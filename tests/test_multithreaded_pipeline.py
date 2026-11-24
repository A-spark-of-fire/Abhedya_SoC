"""
Unit and Smoke Tests for Approach 2 Multi-Threaded Dual Engine with LLM Mutex.
Project: Abhedya SoC (Bharat AI-SoC Challenge - Problem Statement 3)

Verifies:
  1. Concurrent execution of Thread 1 (Live Network NIDS) and Thread 2 (SMS/Phishing Queue).
  2. Threading mutex (llm_lock) prevents simultaneous LLM invocations.
  3. Benign flows and clean SMS messages bypass LLM without acquiring lock (0 LLM compute overhead).
  4. Graceful handling of empty capture windows (IDLE status).
  5. Threat detection triggers ExecuTorch Llama advisory under llm_lock.
  6. Non-blocking queue ingestion under simulated LLM inference.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import threading
import time
import unittest

import pandas as pd

# Configure path resolution
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = WORKSPACE_ROOT / "modules"
for p in [WORKSPACE_ROOT, MODULES_DIR, MODULES_DIR / "sms_spam_detector", MODULES_DIR / "phishguard", MODULES_DIR / "nids"]:
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

from unified_pipeline import (
    UnifiedThreatPipeline,
    UnifiedMultiThreadedEngine,
    llm_lock,
    extract_urls_from_text,
    normalize_sms_text,
)
from llama_prompt_builder import (
    build_llama_prompt,
    build_llama_prompt_pipeline_1,
    build_llama_prompt_pipeline_2,
    generate_grounded_fallback_advisory,
)


class TestMultiThreadedDualEngine(unittest.TestCase):
    """Integration & unit tests for Approach 2 Multi-Threaded Dual Engine."""

    @classmethod
    def setUpClass(cls):
        cls.sample_csv = WORKSPACE_ROOT / "data" / "sample_flows.csv"
        if not cls.sample_csv.exists():
            cls.sample_csv = WORKSPACE_ROOT / "modules" / "nids" / "tests" / "fixtures" / "parity_rows.csv"
        cls.pipeline = UnifiedThreatPipeline(enable_llm_reasoning=True)

    def test_01_concurrent_execution_of_thread1_and_thread2(self):
        """1. Verify concurrent execution of Thread 1 (NIDS) and Thread 2 (SMS/Phishing)."""
        engine = UnifiedMultiThreadedEngine(
            pipeline=self.pipeline,
            capture_window_seconds=1,
        )
        self.assertFalse(engine.is_running)

        engine.start()
        self.assertTrue(engine.is_running)

        # Allow threads to initialize
        time.sleep(1.0)
        status = engine.get_status()

        # Both threads must be concurrently active
        self.assertEqual(status["engine_state"], "RUNNING")
        self.assertIn(status["nids_thread"], ["RUNNING", "CAPTURING", "CLASSIFYING", "IDLE", "CAPTURE_ERROR"])
        self.assertIn(status["sms_thread"], ["RUNNING", "IDLE", "PROCESSING"])

        # Thread 2 handles queue while Thread 1 runs tumbling windows
        engine.submit_sms("Hey, please review the document when you are free.")
        time.sleep(2.0)

        updated_status = engine.get_status()
        self.assertGreaterEqual(updated_status["sms_events_processed"], 1)
        self.assertGreaterEqual(updated_status["nids_windows_processed"], 1)

        engine.stop()
        self.assertFalse(engine.is_running)
        self.assertEqual(engine.get_status()["engine_state"], "STOPPED")

    def test_02_llm_mutex_prevents_simultaneous_invocations(self):
        """2. Verify threading mutex (llm_lock) prevents simultaneous LLM invocations."""
        concurrent_violations = []
        in_critical_section = False
        section_lock = threading.Lock()

        def simulated_llm_work(worker_id: int):
            nonlocal in_critical_section
            for _ in range(5):
                with llm_lock:
                    # Check if another thread is inside the critical section
                    with section_lock:
                        if in_critical_section:
                            concurrent_violations.append(f"Collision detected by worker {worker_id}")
                        in_critical_section = True

                    time.sleep(0.04)  # Simulate ExecuTorch inference time

                    with section_lock:
                        in_critical_section = False

                time.sleep(0.01)

        threads = [
            threading.Thread(target=simulated_llm_work, args=(i,))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(
            len(concurrent_violations), 0,
            f"llm_lock failed to serialize access: {concurrent_violations}"
        )

        # Also test non-blocking acquisition fails when held
        acquired = llm_lock.acquire(blocking=True)
        self.assertTrue(acquired)
        second_acquire = llm_lock.acquire(blocking=False)
        self.assertFalse(second_acquire, "llm_lock must reject secondary acquisition when held")
        llm_lock.release()

    def test_03_benign_flows_and_clean_sms_bypass_llm(self):
        """3. Verify benign flows (P < 0.3545) and clean SMS bypass LLM without acquiring lock."""
        def clean_flow_generator():
            df = pd.read_csv(self.sample_csv)
            # Row 2 is Benign HTTP flow
            return df.iloc[2:3].copy()

        engine = UnifiedMultiThreadedEngine(
            pipeline=self.pipeline,
            capture_window_seconds=1,
            custom_flow_generator=clean_flow_generator,
        )
        engine.start()
        time.sleep(1.0)

        initial_llm_calls = engine.get_status()["llm_invocations"]

        # Submit clean SMS (non-threatening)
        engine.submit_sms("Can we meet at the campus library at 4pm for group study?")
        time.sleep(2.5)

        status = engine.get_status()
        final_llm_calls = status["llm_invocations"]

        # LLM calls MUST remain unchanged for benign events (0 LLM compute overhead)
        self.assertEqual(
            initial_llm_calls, final_llm_calls,
            "Benign flows and clean SMS must bypass LLM engine entirely (0 overhead)"
        )

        # Check logged events
        events = engine.get_event_log(last_n=10)
        sms_events = [e for e in events if e.get("thread") == "sms"]
        self.assertTrue(len(sms_events) > 0)
        self.assertEqual(sms_events[-1]["type"], "CLEAN")
        self.assertFalse(sms_events[-1]["llm_invoked"])
        self.assertLess(sms_events[-1]["processing_latency_ms"], 50.0)

        nids_events = [e for e in events if e.get("thread") == "nids" and e.get("type") == "BENIGN"]
        self.assertTrue(len(nids_events) > 0)
        self.assertEqual(nids_events[-1]["verdict"], "CLEAN")
        self.assertFalse(nids_events[-1]["llm_invoked"])
        self.assertLess(nids_events[-1]["attack_probability"], 0.3545)

        engine.stop()

    def test_04_graceful_handling_of_empty_capture_windows(self):
        """4. Verify graceful handling of empty capture windows (IDLE status)."""
        def empty_window_generator():
            return pd.DataFrame()

        engine = UnifiedMultiThreadedEngine(
            pipeline=self.pipeline,
            capture_window_seconds=1,
            custom_flow_generator=empty_window_generator,
        )
        engine.start()
        time.sleep(2.0)

        status = engine.get_status()
        self.assertEqual(status["engine_state"], "RUNNING")
        self.assertEqual(status["nids_thread"], "IDLE")

        # Verify an IDLE event was logged gracefully without exceptions
        events = engine.get_event_log(last_n=10)
        idle_events = [e for e in events if e.get("thread") == "nids" and e.get("status") == "IDLE"]
        self.assertGreater(len(idle_events), 0, "Engine must log IDLE event for empty capture window")
        self.assertIn("Empty capture window", idle_events[0]["message"])

        engine.stop()
        self.assertEqual(engine.get_status()["engine_state"], "STOPPED")

    def test_05_threat_traffic_triggers_llm_under_mutex(self):
        """5. Verify attack flows and smishing trigger ExecuTorch advisory under llm_lock."""
        engine = UnifiedMultiThreadedEngine(
            pipeline=self.pipeline,
            capture_window_seconds=10,
        )
        engine.start()
        time.sleep(0.5)

        initial_threats = engine.get_status()["total_threats_detected"]
        initial_llm_calls = engine.get_status()["llm_invocations"]

        # Submit known scam message with embedded phishing URL
        engine.submit_sms("URGENT: Your SBI account has been locked. Verify KYC at http://sbi-kyc.top")
        time.sleep(3.5)

        status = engine.get_status()
        self.assertGreater(status["total_threats_detected"], initial_threats)
        self.assertGreater(status["llm_invocations"], initial_llm_calls)

        events = engine.get_event_log(last_n=5)
        threat_events = [e for e in events if e.get("thread") == "sms" and e.get("type") == "THREAT"]
        self.assertGreater(len(threat_events), 0)
        latest = threat_events[-1]
        self.assertTrue(latest["llm_invoked"])
        self.assertIsNotNone(latest.get("advisory"))
        self.assertIn("⚠️ **Threat Summary:**", latest["advisory"])

        engine.stop()

    def test_06_nonblocking_queue_under_simulated_llm_delay(self):
        """6. Verify incoming SMS queue is non-blocking even when LLM is generating text."""
        engine = UnifiedMultiThreadedEngine(
            pipeline=self.pipeline,
            capture_window_seconds=10,
        )
        engine.start()
        time.sleep(0.5)

        # Lock mutex manually to simulate long Llama text generation
        llm_lock.acquire()

        t_submit_start = time.perf_counter()
        # Non-blocking sensor evaluation: submitting should return instantly (<10ms)
        engine.submit_sms("Test message 1")
        engine.submit_sms("Test message 2")
        engine.submit_url("http://example.com")
        submit_duration_ms = (time.perf_counter() - t_submit_start) * 1000.0

        self.assertLess(submit_duration_ms, 25.0, "Submission must be non-blocking even when LLM is busy")
        self.assertGreaterEqual(engine.get_status()["queue_depth"], 0)

        # Release lock so worker threads can resume
        llm_lock.release()
        time.sleep(2.0)

        engine.stop()

    def test_07_explicit_capture_error_reporting(self):
        """7. Verify explicit CAPTURE_ERROR returned when dumpcap is unavailable without silent fallback."""
        res = self.pipeline.run_pipeline_1(live_interface="nonexistent_iface_999", duration_seconds=1)
        self.assertEqual(res["status"], "CAPTURE_ERROR")
        self.assertEqual(res["total_flows_count"], 0)
        self.assertEqual(res["findings"], [])
        self.assertIn("dumpcap unavailable or insufficient permissions (run sudo usermod -aG wireshark $USER)", res["capture_error"])

    def test_08_prompt_safety_and_scammer_advice_prevention(self):
        """8. Verify System Prompt contains negative constraints and advisory prevents scammer echo."""
        from llama_prompt_builder import (
            SYSTEM_PROMPT,
            SYSTEM_PROMPT_PIPELINE_1,
            SYSTEM_PROMPT_PIPELINE_2,
            assemble_advisory,
            SAFETY_NEGATIVE_CONSTRAINT,
        )
        self.assertIn("NEVER tell the user to click links, complete KYC, verify accounts, or call phone numbers", SAFETY_NEGATIVE_CONSTRAINT)
        self.assertIn(SAFETY_NEGATIVE_CONSTRAINT, SYSTEM_PROMPT)
        self.assertIn(SAFETY_NEGATIVE_CONSTRAINT, SYSTEM_PROMPT_PIPELINE_1)
        self.assertIn(SAFETY_NEGATIVE_CONSTRAINT, SYSTEM_PROMPT_PIPELINE_2)

        # Test that scammer advice in generated summary is sanitized
        telemetry = {
            "status": "ALERT",
            "active_threats_count": 1,
            "findings": [
                {
                    "module": "SMS_Spam_Detector",
                    "is_threat": True,
                    "threat_score": 0.99,
                    "threat_level": "HIGH",
                    "detected_triggers": ["kyc", "blocked"]
                }
            ]
        }
        hallucinated_summary = "Please click the link immediately to complete KYC and verify account."
        adv = assemble_advisory(hallucinated_summary, telemetry)
        self.assertNotIn("click the link", adv.lower())
        self.assertIn("⚠️ **Threat Summary:**", adv)
        self.assertIn("🔍 **Technical Reason:**", adv)
        self.assertIn("🛡️ **Recommended Actions:**", adv)

    def test_09_executorch_runner_token_budget_and_stats_filter(self):
        """9. Verify runner default max_new_tokens is 120 and execution stats are filtered."""
        from executorch_runner import ExecuTorchLlamaEngine
        from llama_prompt_builder import filter_executorch_stats
        engine = ExecuTorchLlamaEngine()
        self.assertEqual(engine.max_new_tokens, 120)
        self.assertEqual(engine.timeout_seconds, 180.0)

        raw_stats = "[INFO:llama_runner.cpp:100] Model loaded.\ntokens/sec: 8.5\nprefill time: 1.2s\n⚠️ **Threat Summary:** Active attack detected."
        filtered = filter_executorch_stats(raw_stats)
        self.assertNotIn("tokens/sec", filtered)
        self.assertNotIn("[INFO", filtered)
        self.assertNotIn("prefill time", filtered)
        self.assertIn("Active attack detected", filtered)


if __name__ == "__main__":
    unittest.main()

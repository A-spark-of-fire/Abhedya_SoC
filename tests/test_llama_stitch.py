"""
Unit and Integration Tests for ExecuTorch Llama 3.2 Reasoning Engine & Telemetry Stitching.
Project: Abhedya SoC (Bharat AI-SoC Challenge - Problem Statement 3)
"""

import os
from pathlib import Path
import sys
import unittest

import pandas as pd

# Configure path resolution
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = WORKSPACE_ROOT / "modules"
for p in [WORKSPACE_ROOT, MODULES_DIR, MODULES_DIR / "sms_spam_detector", MODULES_DIR / "phishguard", MODULES_DIR / "nids"]:
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

from llama_prompt_builder import (
    build_llama_prompt,
    generate_grounded_fallback_advisory,
    SYSTEM_PROMPT,
)
from executorch_runner import ExecuTorchLlamaEngine
from unified_pipeline import UnifiedThreatPipeline, extract_urls_from_text


class TestExecuTorchLlamaStitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_csv = WORKSPACE_ROOT / "data" / "sample_flows.csv"
        cls.pipeline = UnifiedThreatPipeline(enable_llm_reasoning=True)

    def test_01_prompt_builder_tokens_and_format(self):
        """Verify prompt builder embeds Llama-3.2 instruct header tokens and JSON telemetry."""
        sample_telemetry = {
            "status": "ALERT",
            "active_threats_count": 2,
            "findings": [
                {
                    "module": "SMS_Spam_Detector",
                    "is_threat": True,
                    "threat_score": 0.985,
                    "threat_level": "HIGH",
                    "detected_triggers": ["urgent", "kyc"]
                },
                {
                    "module": "PhishGuard_URL_Detector",
                    "is_phishing": True,
                    "confidence_score": 0.991,
                    "url": "http://sbi-kyc.top"
                }
            ]
        }
        prompt = build_llama_prompt(sample_telemetry)
        
        # Verify Llama 3.2 special tokens
        self.assertIn("<|begin_of_text|>", prompt)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", prompt)
        self.assertIn("<|eot_id|>", prompt)
        
        # Verify telemetry embedding
        self.assertIn("sbi-kyc.top", prompt)
        self.assertIn("SMS_Spam_Detector", prompt)

    def test_02_deterministic_grounded_fallback_structure(self):
        """Verify fallback generator outputs exact 3-part structured alert."""
        sample_telemetry = {
            "status": "ALERT",
            "active_threats_count": 2,
            "findings": [
                {
                    "module": "SMS_Spam_Detector",
                    "is_threat": True,
                    "threat_score": 0.99,
                    "threat_level": "HIGH",
                    "detected_triggers": ["urgent", "account", "locked"]
                },
                {
                    "module": "PhishGuard_URL_Detector",
                    "is_phishing": True,
                    "confidence_score": 0.97,
                    "url": "http://sbi-kyc.top",
                    "domain_anomalies": ["Suspicious TLD '.top'"]
                }
            ]
        }
        advisory = generate_grounded_fallback_advisory(sample_telemetry)
        
        # Must contain exact 3 section headers
        self.assertIn("⚠️ **Threat Summary:**", advisory)
        self.assertIn("🔍 **Technical Reason:**", advisory)
        self.assertIn("🛡️ **Recommended Actions:**", advisory)
        
        # Exactly 3 numbered action steps
        self.assertIn("1. ", advisory)
        self.assertIn("2. ", advisory)
        self.assertIn("3. ", advisory)
        self.assertNotIn("4. ", advisory)

    def test_03_pipeline_2_smish_plus_phishing_url(self):
        """Verify real workflow for Pipeline 2 (SMS scam with phishing link http://sbi-kyc.top)."""
        sms_text = "URGENT: Your SBI account is blocked. Update KYC immediately at http://sbi-kyc.top"
        
        res = self.pipeline.run_pipeline_2(sms_text)
        
        # Verify status and findings
        self.assertEqual(res["status"], "ALERT")
        self.assertGreaterEqual(res["active_threats_count"], 2)
        
        # Check Auto-Link Extractor worked: findings contain SMS and PhishGuard URL
        modules = [f["module"] for f in res["findings"]]
        self.assertIn("SMS_Spam_Detector", modules)
        self.assertIn("PhishGuard_URL_Detector", modules)
        
        # Verify prompt construction
        self.assertIn("llama_prompt", res)
        self.assertIn("<|begin_of_text|>", res["llama_prompt"])
        self.assertIn("sbi-kyc.top", res["llama_prompt"])
        
        # Verify advisory generation and strict structure
        advisory = res.get("llama_advisory")
        self.assertIsNotNone(advisory)
        self.assertIsInstance(advisory, str)
        self.assertIn("⚠️ **Threat Summary:**", advisory)
        self.assertIn("🔍 **Technical Reason:**", advisory)
        self.assertIn("🛡️ **Recommended Actions:**", advisory)
        self.assertIn("1. ", advisory)
        self.assertIn("2. ", advisory)
        self.assertIn("3. ", advisory)

    def test_04_pipeline_1_real_pcap_replay(self):
        """Verify real PCAP packet replay workflow through Pipeline 1 (Benign flows -> Model Output, LLaMA Bypassed)."""
        sample_pcap = WORKSPACE_ROOT / "data" / "sample_traffic.pcap"
        self.assertTrue(sample_pcap.exists(), f"Sample PCAP missing: {sample_pcap}")
        
        res = self.pipeline.run_pipeline_1(sample_pcap)
        
        self.assertEqual(res["status"], "CLEAN")
        self.assertGreaterEqual(len(res["findings"]), 1)
        self.assertEqual(res["findings"][0]["module"], "Network_NIDS")
        
        # Verify per-flow routing: benign flows are NOT passed to LLaMA -> model output
        self.assertFalse(res["passed_to_llama"])
        self.assertEqual(res["output_type"], "model_output")
        self.assertIsNone(res.get("llama_advisory"))
        self.assertIsNone(res.get("llama_prompt"))
        
        flow0 = res["findings"][0]
        self.assertFalse(flow0["passed_to_llama"])
        self.assertEqual(flow0["output_type"], "model_output")
        self.assertIsNone(flow0["llama_advisory"])
        self.assertIn("verdict", flow0["output"])
        self.assertEqual(flow0["output"]["verdict"], "Benign")

    def test_05_toggle_disable_llm_reasoning(self):
        """Verify that disabling LLM reasoning suppresses advisory while preserving prompt."""
        self.pipeline.enable_llm_reasoning = False
        res = self.pipeline.analyze_sms("URGENT: Bank account locked, verify immediately: http://bad.xyz")
        
        self.assertIn("llama_prompt", res)
        self.assertIsNone(res["llama_advisory"])
        
        # Restore toggle
        self.pipeline.enable_llm_reasoning = True

    def test_06_executorch_engine_safety(self):
        """Verify ExecuTorchLlamaEngine generates advisory without crashing even if binary is absent."""
        engine = ExecuTorchLlamaEngine()
        sample_telemetry = {
            "status": "ALERT",
            "active_threats_count": 1,
            "findings": [
                {
                    "module": "Network_NIDS",
                    "is_attack": True,
                    "attack_type": "DoS_Slowhttptest",
                    "attack_probability": 0.89
                }
            ]
        }
        adv = engine.generate_advisory(sample_telemetry)
        self.assertIsInstance(adv, str)
        self.assertIn("DoS_Slowhttptest", adv)
        self.assertIn("⚠️ **Threat Summary:**", adv)

    def test_07_pipeline_1_and_2_architecture(self):
        """Verify Pipeline 1 per-flow routing (benign -> model output, attack -> llama output) and Pipeline 2."""
        df = pd.read_csv(self.sample_csv)

        # Pipeline 1 - Benign Flow: should NOT pass to LLaMA -> Model Output
        benign_df = df.iloc[2:3].copy()
        res_benign = self.pipeline.run_pipeline_1(benign_df)
        self.assertEqual(res_benign["status"], "CLEAN")
        self.assertFalse(res_benign["passed_to_llama"])
        self.assertEqual(res_benign["output_type"], "model_output")
        self.assertIsNone(res_benign["llama_advisory"])
        self.assertIsNone(res_benign["llama_prompt"])
        self.assertFalse(res_benign["findings"][0]["passed_to_llama"])
        self.assertEqual(res_benign["findings"][0]["output_type"], "model_output")

        # Pipeline 1 - Attack Flow: SHOULD pass to LLaMA -> LLaMA Output
        attack_df = df.iloc[3:4].copy()
        res_p1 = self.pipeline.run_pipeline_1(attack_df)
        self.assertEqual(res_p1["status"], "ALERT")
        self.assertTrue(res_p1["passed_to_llama"])
        self.assertEqual(res_p1["output_type"], "llama_output")
        self.assertTrue(res_p1["findings"][0]["passed_to_llama"])
        self.assertEqual(res_p1["findings"][0]["output_type"], "llama_output")
        self.assertIn("Network_NIDS", res_p1["llama_prompt"])
        self.assertIn("live network intrusion detection (NIDS) telemetry", res_p1["llama_prompt"])
        self.assertIn("NIDS: ~ NIDS telemetry output (for attack)", res_p1["llama_prompt"])
        self.assertIsNotNone(res_p1["llama_advisory"])
        self.assertIn("⚠️ **Threat Summary:**", res_p1["llama_advisory"])

        # Pipeline 2: User Input -> SMS -> Extract URL -> PhishGuard -> Prompt 2
        user_input = "URGENT: Your KYC is expired! Click http://sbi-verification.xyz/update now."
        res_p2 = self.pipeline.run_pipeline_2(user_input)
        self.assertEqual(res_p2["status"], "ALERT")
        self.assertIn("SMS_Spam_Detector", res_p2["llama_prompt"])
        self.assertIn("PhishGuard_URL_Detector", res_p2["llama_prompt"])
        self.assertIn("sbi-verification.xyz", res_p2["llama_prompt"])
        self.assertIn("~ SMS Scam Detector telemetry output ~ PhishGuard URL Detector telemetry output", res_p2["llama_prompt"])
        self.assertIn("⚠️ **Threat Summary:**", res_p2["llama_advisory"])

    def test_08_flask_pipeline_api_endpoints(self):
        """Verify Flask REST endpoints /api/pipeline1 and /api/pipeline2."""
        from app_flask import app
        client = app.test_client()

        # Pipeline 1 API Test (Real PCAP mode - benign flows)
        resp1 = client.post("/api/pipeline1", json={"mode": "pcap"})
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.get_json()
        self.assertIn("pipeline", data1)
        self.assertIn("Pipeline 1", data1["pipeline"])
        self.assertIn("is_attack", data1)
        self.assertIn("passed_to_llama", data1)
        self.assertIn("output_type", data1)
        self.assertFalse(data1["passed_to_llama"])
        self.assertEqual(data1["output_type"], "model_output")

        # Pipeline 1 API Test (Live capture call verification)
        resp_live = client.post("/api/pipeline1", json={"mode": "live", "interface": "lo", "duration": 1, "probe": False})
        self.assertEqual(resp_live.status_code, 200)
        data_live = resp_live.get_json()
        if data_live.get("findings"):
            self.assertIn(data_live["status"], {"CLEAN", "ALERT"})
        else:
            self.assertIn(data_live["status"], {"IDLE", "CAPTURE_ERROR"})
            self.assertEqual(data_live["total_flows_count"], 0)
            self.assertIsNone(data_live["attack_type"])
            self.assertIsNone(data_live["attack_probability"])
            self.assertIsNone(data_live["latency_ms"])
            self.assertIsNone(data_live["flow_info"])

        # Pipeline 2 API Test
        resp2 = client.post("/api/pipeline2", json={
            "user_input": "URGENT: Your bank account is suspended. Verify KYC now at http://sbi-kyc.top"
        })
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertIn("pipeline", data2)
        self.assertIn("Pipeline 2", data2["pipeline"])
        self.assertIn("sms_telemetry", data2)
        self.assertIn("extracted_urls", data2)
        self.assertEqual(data2["status"], "ALERT")
        self.assertIn("llama_prompt", data2)
        self.assertIn("llama_advisory", data2)

    def test_09_pipeline_1_live_interface_call(self):
        """Verify Pipeline 1 accepts live_interface and duration_seconds keyword arguments.

        A live window with no traffic must return an explicit IDLE (or
        CAPTURE_ERROR) status instead of a fabricated benign ``idle-window``
        flow.
        """
        res = self.pipeline.run_pipeline_1(live_interface="lo", duration_seconds=1)
        self.assertIn("status", res)
        self.assertIn("findings", res)
        if res["findings"]:
            self.assertEqual(res["findings"][0]["module"], "Network_NIDS")
            self.assertIn(res["status"], {"ALERT", "CLEAN"})
        else:
            self.assertIn(res["status"], {"IDLE", "CAPTURE_ERROR"})
            self.assertEqual(res["total_flows_count"], 0)
            self.assertFalse(res["passed_to_llama"])
            self.assertEqual(res["output_type"], "no_output")

    def test_10_pipeline_1_window_per_flow_routing(self):
        """Verify per-flow routing in a mixed window: benign -> model output, attack -> llama output."""
        df = pd.read_csv(self.sample_csv)
        # Row 2 is Benign; Row 3 is DoS_Slowhttptest (Attack)
        mixed_window_df = df.iloc[2:4].copy()
        res = self.pipeline.run_pipeline_1(mixed_window_df)
        
        self.assertEqual(res["status"], "ALERT")
        self.assertEqual(res["active_threats_count"], 1)
        self.assertEqual(res["total_flows_count"], 2)
        self.assertTrue(res["passed_to_llama"])
        self.assertEqual(res["output_type"], "llama_output")
        
        findings = res["findings"]
        self.assertEqual(len(findings), 2)
        
        # Flow 0: Benign flow
        f_benign = findings[0]
        self.assertFalse(f_benign["is_attack"])
        self.assertFalse(f_benign["passed_to_llama"])
        self.assertEqual(f_benign["output_type"], "model_output")
        self.assertIsNone(f_benign["llama_prompt"])
        self.assertIsNone(f_benign["llama_advisory"])
        self.assertEqual(f_benign["output"]["verdict"], "Benign")
        
        # Flow 1: Attack flow
        f_attack = findings[1]
        self.assertTrue(f_attack["is_attack"])
        self.assertTrue(f_attack["passed_to_llama"])
        self.assertEqual(f_attack["output_type"], "llama_output")
        self.assertIsNotNone(f_attack["llama_prompt"])
        self.assertIsNotNone(f_attack["llama_advisory"])
        self.assertIn("DoS_Slowhttptest", f_attack["llama_prompt"])

    def test_11_empty_window_telemetry_schema(self):
        """Verify empty window contract: status IDLE, findings [], output_type no_output, no fake idle-window."""
        empty_df = pd.DataFrame()
        res = self.pipeline.run_pipeline_1(empty_df)
        self.assertEqual(res["status"], "IDLE")
        self.assertEqual(res["total_flows_count"], 0)
        self.assertEqual(res["active_threats_count"], 0)
        self.assertEqual(res["findings"], [])
        self.assertFalse(res["passed_to_llama"])
        self.assertEqual(res["output_type"], "no_output")
        self.assertIsNone(res["llama_prompt"])
        self.assertIsNone(res["llama_advisory"])
        self.assertEqual(res["output"]["verdict"], "NO_FLOWS")

    def test_12_pipeline_1_default_window_size_1min(self):
        """Verify Pipeline 1 default window size is 1 min (60 seconds)."""
        empty_df = pd.DataFrame()
        res = self.pipeline.run_pipeline_1(empty_df)
        self.assertEqual(res.get("window_size_seconds"), 60)
        self.assertEqual(res.get("window_size_label"), "1 min")

    def test_13_pipeline_1_stream_generator_discards_previous(self):
        """Verify continuous stream generator yields 1-min windows and updates window metadata."""
        stream = self.pipeline.stream_pipeline_1(live_interface="lo", duration_seconds=1, max_windows=2)
        windows = list(stream)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["window_number"], 1)
        self.assertEqual(windows[1]["window_number"], 2)
        self.assertEqual(windows[0]["window_size_seconds"], 1)
        self.assertEqual(windows[1]["window_size_seconds"], 1)

    def test_14_pipeline_1_flask_api_default_window(self):
        """Verify Flask API /api/pipeline1 endpoint defaults to 60s (1 min) window size."""
        from app_flask import app
        client = app.test_client()
        resp = client.post("/api/pipeline1", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("window_size_seconds"), 60)
        self.assertEqual(data.get("window_size_label"), "1 min")


if __name__ == "__main__":
    unittest.main()

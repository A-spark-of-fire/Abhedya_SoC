"""
Unit and Integration Tests for ExecuTorch Llama 3.2 Reasoning Engine & Telemetry Stitching.
Project: Abhedya SoC / Trinetra AI (Bharat AI-SoC Challenge - Problem Statement 3)
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

    def test_03_unified_pipeline_smish_plus_dos_flow(self):
        """Simulate an SMS scam with phishing link (http://sbi-kyc.top) + NIDS DoS attack flow."""
        sms_text = "URGENT: Your SBI account is blocked. Update KYC immediately at http://sbi-kyc.top"
        
        # Load DoS or attack flow from sample CSV
        self.assertTrue(self.sample_csv.exists(), f"Sample CSV missing: {self.sample_csv}")
        df = pd.read_csv(self.sample_csv)
        # Row 3 is DoS_Slowhttptest attack in sample_flows.csv
        flow_df = df.iloc[3:4].copy()
        
        res = self.pipeline.analyze_unified_event(sms_text=sms_text, flow_df=flow_df)
        
        # Verify status and findings
        self.assertEqual(res["status"], "ALERT")
        self.assertGreaterEqual(res["active_threats_count"], 2)
        
        # Check Auto-Link Extractor worked: 3 findings (SMS, PhishGuard URL, NIDS)
        modules = [f["module"] for f in res["findings"]]
        self.assertIn("SMS_Spam_Detector", modules)
        self.assertIn("PhishGuard_URL_Detector", modules)
        self.assertIn("Network_NIDS", modules)
        
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

    def test_04_clean_traffic_and_sms(self):
        """Verify pipeline returns CLEAN and benign advisory for non-threatening input."""
        sms_text = "Hello, can we schedule the project sync meeting tomorrow at 10 AM?"
        df = pd.read_csv(self.sample_csv)
        # Row 2 is Benign HTTP flow
        flow_df = df.iloc[2:3].copy()
        
        res = self.pipeline.analyze_unified_event(sms_text=sms_text, flow_df=flow_df)
        
        self.assertEqual(res["status"], "CLEAN")
        self.assertEqual(res["active_threats_count"], 0)
        
        advisory = res.get("llama_advisory")
        self.assertIsNotNone(advisory)
        self.assertIn("⚠️ **Threat Summary:**", advisory)
        self.assertIn("No active threats were detected", advisory)

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


if __name__ == "__main__":
    unittest.main()

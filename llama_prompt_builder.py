"""
Prompt Constructor Engine for Abhedya SoC / Trinetra AI
Bharat AI-SoC Challenge — Problem Statement 3: On-Device Security

Translates Tier 1 multi-model threat telemetry (SMS, PhishGuard, Network NIDS)
into an on-device zero-shot prompt formatted for Meta Llama 3.2 ExecuTorch (1B-Instruct).
Also provides a deterministic rule-based template advisory generator for robust fallback.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = (
    "You are the on-device security assistant for Abhedya SoC / Trinetra AI.\n"
    "Your job is to translate technical threat telemetry from 3 on-device ML sensors "
    "(SMS Scam Detector, PhishGuard URL Analyzer, and Network NIDS) into a simple, "
    "plain-English advisory for non-technical users.\n\n"
    "STRICT RULES:\n"
    "1. Ground your answer strictly in the provided JSON telemetry. Never invent threats not present in the payload.\n"
    "2. Output a 3-part structured alert:\n"
    "   - ⚠️ **Threat Summary:** What was detected in plain English.\n"
    "   - 🔍 **Technical Reason:** Why it is dangerous (explain SHAP metrics, suspicious TLDs, or SMS trigger words simply).\n"
    "   - 🛡️ **Recommended Actions:** Exactly 3 clear safety steps the user should take."
)


def format_telemetry_for_prompt(telemetry_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes and compacts the telemetry payload before feeding it to the LLM prompt.
    Strips raw prompt recursion and non-essential binary keys.
    """
    clean_dict: Dict[str, Any] = {
        "status": telemetry_json.get("status", "ALERT"),
        "timestamp": telemetry_json.get("timestamp"),
        "active_threats_count": telemetry_json.get("active_threats_count", 0),
        "findings": []
    }

    for f in telemetry_json.get("findings", []):
        mod = f.get("module")
        if mod == "SMS_Spam_Detector":
            clean_dict["findings"].append({
                "module": mod,
                "raw_text": f.get("raw_text"),
                "is_threat": f.get("is_threat"),
                "threat_score": f.get("threat_score"),
                "threat_level": f.get("threat_level"),
                "detected_triggers": f.get("detected_triggers", [])
            })
        elif mod == "PhishGuard_URL_Detector":
            clean_dict["findings"].append({
                "module": mod,
                "url": f.get("url"),
                "is_phishing": f.get("is_phishing"),
                "confidence_score": f.get("confidence_score"),
                "domain_anomalies": f.get("domain_anomalies", [])
            })
        elif mod == "Network_NIDS":
            clean_dict["findings"].append({
                "module": mod,
                "flow_id": f.get("flow_id"),
                "src_ip": f.get("src_ip"),
                "dst_ip": f.get("dst_ip"),
                "dst_port": f.get("dst_port"),
                "protocol": f.get("protocol"),
                "is_attack": f.get("is_attack"),
                "attack_type": f.get("attack_type"),
                "confidence": f.get("confidence"),
                "shap_evidence": f.get("shap_evidence", [])
            })

    return clean_dict


def build_llama_prompt(telemetry_json: Dict[str, Any]) -> str:
    """
    Constructs the zero-shot Llama-3.2 Chat prompt matching ExecuTorch token contract:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> ... <|eot_id|>
    <|start_header_id|>user<|end_header_id|> ... <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    clean_telemetry = format_telemetry_for_prompt(telemetry_json)
    json_str = json.dumps(clean_telemetry, indent=2)

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Review the following on-device threat telemetry and generate the plain-language safety alert:\n\n"
        "```json\n"
        f"{json_str}\n"
        "```\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt


def generate_grounded_fallback_advisory(telemetry_json: Dict[str, Any]) -> str:
    """
    Deterministic rule-based advisory generator grounded strictly in the telemetry findings.
    Used when local ExecuTorch C++ binary is not running, times out, or when offline.
    Follows the exact 3-part structured alert specification.
    """
    status = telemetry_json.get("status", "CLEAN")
    active_count = telemetry_json.get("active_threats_count", 0)
    findings = telemetry_json.get("findings", [])

    if status == "CLEAN" or active_count == 0:
        return (
            "⚠️ **Threat Summary:** No active threats were detected on your device. "
            "All incoming messages, web links, and network flows are within normal parameters.\n\n"
            "🔍 **Technical Reason:** The on-device SMS, PhishGuard, and Network intrusion sensors "
            "evaluated all data against trained security baselines and found zero malicious indicators.\n\n"
            "🛡️ **Recommended Actions:**\n"
            "1. No immediate action required; your device is currently secure.\n"
            "2. Continue monitoring incoming communications normally.\n"
            "3. Keep your on-device threat detection active."
        )

    # 1. Threat Summary Breakdown
    summary_parts = []
    reason_parts = []
    action_parts = []

    has_sms_threat = False
    has_phish_threat = False
    has_nids_threat = False

    for f in findings:
        mod = f.get("module")
        if mod == "SMS_Spam_Detector" and f.get("is_threat"):
            has_sms_threat = True
            level = f.get("threat_level", "HIGH")
            triggers = f.get("detected_triggers", [])
            trigger_str = f"keyword triggers ({', '.join(triggers)})" if triggers else "social engineering patterns"
            summary_parts.append(f"A suspicious SMS message ({level} risk) attempting to impersonate a legitimate service was received.")
            reason_parts.append(
                f"The message contains high-urgency {trigger_str} and text obfuscation designed to bypass standard filters."
            )

        elif mod == "PhishGuard_URL_Detector" and f.get("is_phishing"):
            has_phish_threat = True
            conf = f.get("confidence_score", 0.0) * 100.0
            url = f.get("url", "unknown link")
            anomalies = f.get("domain_anomalies", [])
            summary_parts.append(f"A deceptive phishing website link ({url[:45]}...) was identified with {conf:.1f}% confidence.")
            anomaly_summary = "; ".join(anomalies[:2]) if anomalies else "suspicious structural domain features"
            reason_parts.append(
                f"The web link exhibits high-risk indicators: {anomaly_summary}."
            )

        elif mod == "Network_NIDS" and f.get("is_attack"):
            has_nids_threat = True
            attack_type = f.get("attack_type", "Network Intrusion")
            src_ip = f.get("src_ip", "remote host")
            dst_port = f.get("dst_port", 0)
            shap_evidence = f.get("shap_evidence", [])
            top_feature = shap_evidence[0]["feature"] if shap_evidence else "packet burst signature"

            summary_parts.append(f"An active network intrusion ({attack_type}) originating from IP {src_ip} targeting port {dst_port} was detected.")
            reason_parts.append(
                f"Mathematical evidence shows abnormal flow traffic patterns (primarily driven by '{top_feature}') consistent with unauthorized network attack activity."
            )

    # Build 3 clear recommended actions based on active threat vectors
    if has_sms_threat and has_phish_threat and has_nids_threat:
        action_parts = [
            "Do NOT open, reply to, or click any links inside the flagged SMS message.",
            "Do NOT enter your bank credentials, passwords, or personal details on the flagged website.",
            "Disconnect from the current untrusted network or isolate the device while edge firewall rules block the attacking IP."
        ]
    elif has_sms_threat or has_phish_threat:
        action_parts = [
            "Do NOT click the web link or call any phone numbers listed in the message.",
            "Delete the fraudulent SMS immediately to avoid accidental interaction.",
            "Verify your account status only through your bank's official mobile application or official website."
        ]
    elif has_nids_threat:
        action_parts = [
            "Block incoming connections from the flagged source IP using the local edge firewall.",
            "Verify that open ports on your device are protected and update system access credentials.",
            "Inspect running network services and terminate any unauthorized background processes."
        ]
    else:
        action_parts = [
            "Exercise caution before clicking any unverified links.",
            "Do not disclose confidential personal or banking credentials.",
            "Keep your edge security system active and updated."
        ]

    # Combine into 3-part structured alert
    summary_text = " ".join(summary_parts) if summary_parts else f"{active_count} security risk vector(s) were flagged by local sensors."
    reason_text = " ".join(reason_parts) if reason_parts else "Multiple ML sensor anomalies crossed configured security thresholds."

    return (
        f"⚠️ **Threat Summary:** {summary_text}\n\n"
        f"🔍 **Technical Reason:** {reason_text}\n\n"
        f"🛡️ **Recommended Actions:**\n"
        f"1. {action_parts[0]}\n"
        f"2. {action_parts[1]}\n"
        f"3. {action_parts[2]}"
    )


def clean_llm_response(raw_output: str) -> str:
    """Strips ExecuTorch stop tokens and extra whitespace."""
    if not raw_output:
        return ""
    text = str(raw_output).strip()
    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]
    for tok in stop_tokens:
        text = text.replace(tok, "")
    return text.strip()

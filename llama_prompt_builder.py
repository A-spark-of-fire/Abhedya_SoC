"""
Prompt Constructor Engine for Abhedya SoC
Bharat AI-SoC Challenge — Problem Statement 3: On-Device Security

Translates Tier 1 multi-model threat telemetry (SMS, PhishGuard, Network NIDS)
into an on-device zero-shot prompt formatted for Meta Llama 3.2 ExecuTorch (1B-Instruct).
Enforces prompt safety constraints:
- Llama 3.2 is constrained to write ONLY the 1-sentence plain-English threat summary.
- The Technical Reason and 3 Recommended Safety Actions are populated from
  pre-reviewed, deterministic safe text templates (SHAP attributions & trigger dictionary).
- Explicit negative constraints prevent echoing scammer/attacker instructions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

# Negative safety constraint enforced across all System Prompts (Problem 2)
SAFETY_NEGATIVE_CONSTRAINT = (
    "NEVER tell the user to click links, complete KYC, verify accounts, or call phone numbers present in the threat payload. "
    "ALWAYS instruct the user to ignore deceptive requests, block senders, and contact official channels independently."
)

SYSTEM_PROMPT = (
    "You are the on-device security assistant for Abhedya SoC.\n"
    "Your job is to translate technical threat telemetry from 3 on-device ML sensors "
    "(SMS Scam Detector, PhishGuard URL Analyzer, and Network NIDS) into a simple, "
    "plain-English threat summary sentence for non-technical users.\n\n"
    "STRICT RULES:\n"
    "1. Ground your answer strictly in the provided JSON telemetry. Never invent threats not present in the payload.\n"
    "2. Write ONLY a single, plain-English threat summary sentence describing what malicious activity was detected.\n"
    f"3. {SAFETY_NEGATIVE_CONSTRAINT}\n"
    "4. Do NOT output technical reasons or recommended safety actions. Output ONLY the 1-sentence plain-English threat summary."
)

# Pipeline 1: Live Network -> NIDS -> Llama Prompt 1 (Input: NIDS telemetry for attack) -> Output
SYSTEM_PROMPT_PIPELINE_1 = (
    "You are the on-device network security assistant for Abhedya SoC.\n"
    "Your job is to translate technical live network intrusion detection telemetry from the on-device NIDS "
    "(13-class XGBoost classifier and SHAP mathematical explainability engine) into a simple, "
    "plain-English threat summary sentence for security operators and non-technical users.\n\n"
    "STRICT RULES:\n"
    "1. Ground your answer strictly in the provided NIDS telemetry. Never invent attacks or IP addresses not present in the payload.\n"
    "2. Write ONLY a single, plain-English threat summary sentence describing what network intrusion was detected.\n"
    f"3. {SAFETY_NEGATIVE_CONSTRAINT}\n"
    "4. Do NOT output technical reasons or recommended safety actions. Output ONLY the 1-sentence plain-English threat summary."
)

# Pipeline 2: User Input -> SMS Scam Detector -> Extract URL -> PhishGuard URL Detector -> Llama Prompt 2 -> Output
SYSTEM_PROMPT_PIPELINE_2 = (
    "You are the on-device communication & web security assistant for Abhedya SoC.\n"
    "Your job is to translate technical threat telemetry from user input analysis—correlating the on-device "
    "SMS Scam Detector and the PhishGuard URL Detector—into a simple, plain-English threat summary sentence for end users.\n\n"
    "STRICT RULES:\n"
    "1. Ground your answer strictly in the provided SMS and PhishGuard URL telemetry. Never invent threats or links not present in the payload.\n"
    "2. Write ONLY a single, plain-English threat summary sentence describing what deceptive scam or phishing attempt was detected.\n"
    f"3. {SAFETY_NEGATIVE_CONSTRAINT}\n"
    "4. Do NOT output technical reasons or recommended safety actions. Output ONLY the 1-sentence plain-English threat summary."
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


def build_llama_prompt_pipeline_1(nids_telemetry: Dict[str, Any]) -> str:
    """
    Pipeline 1 Prompt Constructor:
    Live Network -> NIDS -> Llama Prompt 1 (Input: NIDS telemetry for attack) -> Output.
    Constrains Llama 3.2 to write ONLY the 1-sentence plain-English threat summary.
    """
    clean_telemetry = format_telemetry_for_prompt(nids_telemetry)
    attack_findings = [f for f in clean_telemetry.get("findings", []) if f.get("is_attack")]
    telemetry_to_prompt = dict(clean_telemetry)
    if attack_findings:
        telemetry_to_prompt["nids_attack_telemetry"] = attack_findings
    json_str = json.dumps(telemetry_to_prompt, indent=2)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT_PIPELINE_1}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Review the following live network intrusion detection (NIDS) telemetry output (for attack) and write ONLY a 1-sentence plain-language threat summary:\n\n"
        "Input: NIDS: ~ NIDS telemetry output (for attack)\n"
        "```json\n"
        f"{json_str}\n"
        "```\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

# Aliases for Pipeline 1
build_prompt_1_nids = build_llama_prompt_pipeline_1
build_llama_prompt_1 = build_llama_prompt_pipeline_1


def build_llama_prompt_pipeline_2(
    sms_telemetry: Dict[str, Any],
    url_telemetry: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
) -> str:
    """
    Pipeline 2 Prompt Constructor:
    User Input -> SMS Scam Detector -> Extract URL -> PhishGuard URL Detector -> Llama Prompt 2 -> Output.
    Constrains Llama 3.2 to write ONLY the 1-sentence plain-English threat summary.
    """
    if "findings" in sms_telemetry:
        combined_payload = dict(sms_telemetry)
    else:
        findings = [sms_telemetry]
        if url_telemetry:
            if isinstance(url_telemetry, list):
                findings.extend(url_telemetry)
            else:
                findings.append(url_telemetry)
        active_count = sum(1 for f in findings if f.get("is_threat") or f.get("is_phishing"))
        combined_payload = {
            "status": "ALERT" if active_count > 0 else "CLEAN",
            "active_threats_count": active_count,
            "findings": findings
        }

    clean_telemetry = format_telemetry_for_prompt(combined_payload)
    json_str = json.dumps(clean_telemetry, indent=2)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT_PIPELINE_2}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Review the following user input threat telemetry (~ SMS Scam Detector telemetry output ~ PhishGuard URL Detector telemetry output) and write ONLY a 1-sentence plain-language threat summary:\n\n"
        "Input: ~ SMS Scam Detector telemetry output ~ PhishGuard URL Detector telemetry output\n"
        "```json\n"
        f"{json_str}\n"
        "```\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

# Aliases for Pipeline 2
build_prompt_2_smishing = build_llama_prompt_pipeline_2
build_llama_prompt_2 = build_llama_prompt_pipeline_2


def build_llama_prompt(telemetry_json: Dict[str, Any], pipeline_type: Optional[int] = None) -> str:
    """
    Constructs the zero-shot Llama-3.2 Chat prompt matching ExecuTorch token contract:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> ... <|eot_id|>
    <|start_header_id|>user<|end_header_id|> ... <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    findings = telemetry_json.get("findings", [])
    modules_present = {f.get("module") for f in findings if f.get("module")}

    if pipeline_type == 1 or (modules_present == {"Network_NIDS"}):
        return build_llama_prompt_pipeline_1(telemetry_json)
    elif pipeline_type == 2 or (modules_present and modules_present.issubset({"SMS_Spam_Detector", "PhishGuard_URL_Detector"})):
        return build_llama_prompt_pipeline_2(telemetry_json)

    clean_telemetry = format_telemetry_for_prompt(telemetry_json)
    json_str = json.dumps(clean_telemetry, indent=2)

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Review the following on-device threat telemetry and write ONLY a 1-sentence plain-language threat summary:\n\n"
        "```json\n"
        f"{json_str}\n"
        "```\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


# =====================================================================
# DETERMINISTIC SAFE REMEDIATION & TEMPLATES (PROBLEM 2)
# =====================================================================

DANGEROUS_HALLUCINATION_PATTERNS = [
    r"\bclick\s+(?:the\s+|this\s+)?link\b",
    r"\bcomplete\s+kyc\b",
    r"\bverify\s+(?:your\s+)?account\b",
    r"\bupdate\s+kyc\b",
    r"\bcall\s+(?:the\s+)?(?:phone\s+)?number\b",
    r"\bcall\s+\+?\d{5,}\b",
    r"\blogin\s+to\s+verify\b",
    r"\benter\s+(?:your\s+)?(?:bank\s+)?credentials\b",
    r"\bprovide\s+otp\b",
]


def sanitize_threat_summary(raw_summary: str, default_summary: str) -> str:
    """
    Ensures the generated threat summary is safe, single-sentence, and free
    of echoed scammer advice.
    """
    if not raw_summary:
        return default_summary

    text = raw_summary.strip()
    text = re.sub(r'^(?:⚠️\s*)?\*?\*?threat\s+summary:\*?\*?\s*', '', text, flags=re.IGNORECASE).strip()

    # Reject summary if it echoes attacker/scammer instructions
    for pat in DANGEROUS_HALLUCINATION_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return default_summary

    # Ensure single concise sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    first_sentence = sentences[0].strip() if sentences else text
    if len(first_sentence) < 15:
        return default_summary
    return first_sentence


def get_deterministic_technical_reason(telemetry_json: Dict[str, Any]) -> str:
    """
    Populates Technical Reason deterministically from SMS/PhishGuard trigger dictionary
    and NIDS SHAP feature attributions.
    """
    findings = telemetry_json.get("findings", [])
    reason_parts = []

    for f in findings:
        mod = f.get("module")
        if mod == "SMS_Spam_Detector" and f.get("is_threat"):
            triggers = f.get("detected_triggers", [])
            trigger_str = f"keyword triggers ({', '.join(triggers)})" if triggers else "social engineering urgency patterns"
            reason_parts.append(
                f"The message contains high-urgency {trigger_str} and text obfuscation designed to bypass standard filters."
            )
        elif mod == "PhishGuard_URL_Detector" and f.get("is_phishing"):
            anomalies = f.get("domain_anomalies", [])
            anomaly_summary = "; ".join(anomalies[:2]) if anomalies else "suspicious structural domain features"
            reason_parts.append(
                f"The web link exhibits high-risk indicators: {anomaly_summary}."
            )
        elif mod == "Network_NIDS" and f.get("is_attack"):
            shap_evidence = f.get("shap_evidence", [])
            top_feature = shap_evidence[0]["feature"] if shap_evidence else "packet burst signature"
            reason_parts.append(
                f"Mathematical evidence shows abnormal flow traffic patterns (primarily driven by '{top_feature}') consistent with unauthorized network attack activity."
            )

    if reason_parts:
        return " ".join(reason_parts)
    return "Multiple ML sensor anomalies crossed configured security thresholds."


def get_deterministic_recommended_actions(telemetry_json: Dict[str, Any]) -> List[str]:
    """
    Populates exactly 3 recommended safety actions from pre-reviewed, deterministic safe text templates.
    """
    findings = telemetry_json.get("findings", [])
    has_sms = any(f.get("module") == "SMS_Spam_Detector" and f.get("is_threat") for f in findings)
    has_phish = any(f.get("module") == "PhishGuard_URL_Detector" and f.get("is_phishing") for f in findings)
    has_nids = any(f.get("module") == "Network_NIDS" and f.get("is_attack") for f in findings)

    if (has_sms or has_phish) and has_nids:
        return [
            "Do NOT open, reply to, or click any links inside the flagged SMS message.",
            "Do NOT enter your bank credentials, passwords, or personal details on the flagged website.",
            "Disconnect from the current untrusted network or isolate the device while edge firewall rules block the attacking IP."
        ]
    elif has_sms or has_phish:
        return [
            "Do NOT click the web link or call any phone numbers listed in the message.",
            "Delete the fraudulent SMS immediately to avoid accidental interaction.",
            "Verify your account status only through your bank's official mobile application or official website."
        ]
    elif has_nids:
        return [
            "Block incoming connections from the flagged source IP using the local edge firewall.",
            "Verify that open ports on your device are protected and update system access credentials.",
            "Inspect running network services and terminate any unauthorized background processes."
        ]
    else:
        return [
            "No immediate action required; your device is currently secure.",
            "Continue monitoring incoming communications normally.",
            "Keep your on-device threat detection active."
        ]


def build_default_threat_summary(telemetry_json: Dict[str, Any]) -> str:
    """Builds a deterministic default threat summary sentence from findings."""
    status = telemetry_json.get("status", "CLEAN")
    active_count = telemetry_json.get("active_threats_count", 0)
    findings = telemetry_json.get("findings", [])

    if status == "CLEAN" or active_count == 0:
        return "No active threats were detected on your device; all incoming communications and network flows are within normal parameters."

    summary_parts = []
    for f in findings:
        mod = f.get("module")
        if mod == "SMS_Spam_Detector" and f.get("is_threat"):
            level = f.get("threat_level", "HIGH")
            summary_parts.append(f"A suspicious SMS message ({level} risk) attempting to impersonate a legitimate service was received.")
        elif mod == "PhishGuard_URL_Detector" and f.get("is_phishing"):
            conf = f.get("confidence_score", 0.0) * 100.0
            url = f.get("url", "unknown link")
            summary_parts.append(f"A deceptive phishing website link ({url[:45]}...) was identified with {conf:.1f}% confidence.")
        elif mod == "Network_NIDS" and f.get("is_attack"):
            attack_type = f.get("attack_type", "Network Intrusion")
            src_ip = f.get("src_ip", "remote host")
            dst_port = f.get("dst_port", 0)
            summary_parts.append(f"An active network intrusion ({attack_type}) originating from IP {src_ip} targeting port {dst_port} was detected.")

    if summary_parts:
        return " ".join(summary_parts)
    return f"{active_count} security risk vector(s) were flagged by local sensors."


def assemble_advisory(threat_summary: Optional[str], telemetry_json: Dict[str, Any]) -> str:
    """
    Enforces the safe remediation contract:
    - ⚠️ Threat Summary: 1-sentence plain-English summary (from Llama 3.2 if safe, else deterministic).
    - 🔍 Technical Reason: Deterministically generated from SMS triggers, PhishGuard anomalies, and NIDS SHAP.
    - 🛡️ Recommended Actions: Exactly 3 safe steps from pre-reviewed deterministic templates.
    """
    default_summary = build_default_threat_summary(telemetry_json)
    safe_summary = sanitize_threat_summary(threat_summary or "", default_summary)
    technical_reason = get_deterministic_technical_reason(telemetry_json)
    actions = get_deterministic_recommended_actions(telemetry_json)

    return (
        f"⚠️ **Threat Summary:** {safe_summary}\n\n"
        f"🔍 **Technical Reason:** {technical_reason}\n\n"
        f"🛡️ **Recommended Actions:**\n"
        f"1. {actions[0]}\n"
        f"2. {actions[1]}\n"
        f"3. {actions[2]}"
    )


def generate_grounded_fallback_advisory(telemetry_json: Dict[str, Any]) -> str:
    """
    Deterministic rule-based advisory generator grounded strictly in the telemetry findings.
    Used when local ExecuTorch C++ binary is not running, times out, or when offline.
    Follows the exact 3-part structured alert specification.
    """
    status = telemetry_json.get("status", "CLEAN")
    active_count = telemetry_json.get("active_threats_count", 0)

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

    return assemble_advisory(None, telemetry_json)


def filter_executorch_stats(text: str) -> str:
    """Filter out stdout execution statistics lines emitted by llama_main (e.g. [INFO], tokens/sec, prefill time)."""
    if not text:
        return ""
    stats_keywords = [
        "[info",
        "tokens/sec",
        "tokens / sec",
        "prefill time",
        "eval time",
        "model load time",
        "prompt eval",
        "sampling time",
        "total time",
        "throughput",
        "generated tokens",
        "tokens per sec",
    ]
    lines = []
    for line in text.splitlines():
        line_stripped = line.strip()
        lower_line = line_stripped.lower()
        if any(kw in lower_line for kw in stats_keywords):
            continue
        if re.match(r"^[IWEF]\d{4}\s", line_stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def clean_llm_response(raw_output: str) -> str:
    """Strips ExecuTorch stop tokens, execution statistics lines, and extra whitespace."""
    if not raw_output:
        return ""
    text = str(raw_output).strip()
    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>"]
    for tok in stop_tokens:
        text = text.replace(tok, "")
    return filter_executorch_stats(text)

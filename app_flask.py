"""
Bharat AI-SoC Challenge - Problem Statement 3: On-Device Edge Security
Lightweight Flask + HTML5/CSS3/JS Web Interface & Telemetry API Server
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent
MODULES_DIR = WORKSPACE_ROOT / "modules"
for p in [WORKSPACE_ROOT, MODULES_DIR, MODULES_DIR / "sms_spam_detector", MODULES_DIR / "phishguard", MODULES_DIR / "nids"]:
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

from unified_pipeline import (
    UnifiedThreatPipeline,
    extract_urls_from_text,
    normalize_sms_text,
    generate_llama_executorch_prompt,
    get_dumpcap_info,
    get_available_interfaces,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize pipeline lazily or on startup
pipeline: UnifiedThreatPipeline | None = None

def get_pipeline() -> UnifiedThreatPipeline:
    global pipeline
    if pipeline is None:
        pipeline = UnifiedThreatPipeline()
    return pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ONLINE",
        "target_hardware": "Raspberry Pi 5 / Bharat AI-SoC",
        "pipelines": {
            "pipeline_1": "Live Network -> NIDS -> Llama Prompt 1 (NIDS Telemetry for Attack) -> Output",
            "pipeline_2": "User Input -> SMS Scam Detector -> Extract URL -> PhishGuard URL Detector -> Llama Prompt 2 -> Output"
        },
        "edge_models": {
            "sms_scam_detector": "ACTIVE (Stacking Ensemble, 25k dims, tau=0.390)",
            "phishguard_url": "ACTIVE (PyTorch MLP, 375k params)",
            "network_nids": "ACTIVE (13-Class XGBoost + SHAP, tau=0.3545)"
        },
        "downstream_engine": "Meta Llama 3.2 ExecuTorch 1B",
        "live_capture": get_dumpcap_info(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/api/interfaces", methods=["GET"])
def get_interfaces():
    """Returns detected edge network interfaces for live capture."""
    return jsonify({"interfaces": get_available_interfaces()})


@app.route("/api/pipeline1", methods=["POST"])
def run_pipeline_1_endpoint():
    """
    Pipeline 1 Endpoint:
    Live Network (dumpcap / PCAP / Flow records) -> NIDS -> Llama Prompt 1 -> Output Advisory
    Standard Window Size: 1 min (60 seconds)
    """
    data = request.get_json(force=True) if request.data else {}
    mode = data.get("mode", "pcap")
    interface = data.get("interface", "any")
    duration = int(data.get("duration", 60))
    window_number = int(data.get("window_number", 1))
    enable_llm = data.get("enable_llm_reasoning", True)
    flow_data = data.get("flow")

    pipe = get_pipeline()
    pipe.enable_llm_reasoning = bool(enable_llm)

    probe = bool(data.get("probe", False))
    if mode == "live" or data.get("live"):
        telemetry = pipe.run_pipeline_1(live_interface=interface, duration_seconds=duration, probe=probe)
    elif flow_data:
        telemetry = pipe.run_pipeline_1(flow_data=pd.DataFrame([flow_data]) if isinstance(flow_data, dict) else flow_data)
    else:
        sample_pcap = WORKSPACE_ROOT / "data" / "sample_traffic.pcap"
        telemetry = pipe.run_pipeline_1(flow_data=sample_pcap)

    finding0 = telemetry["findings"][0] if telemetry["findings"] else None

    response = {
        "pipeline": "Pipeline 1: Live Network -> NIDS -> Per-Flow Routing (Benign: Model Output | Attack: LLaMA Output)",
        "status": telemetry["status"],
        "timestamp": telemetry["timestamp"],
        "window_size_seconds": duration,
        "window_size_label": f"{duration // 60} min" if duration % 60 == 0 else f"{duration}s",
        "window_number": window_number,
        "active_threats_count": telemetry["active_threats_count"],
        "total_flows_count": telemetry.get("total_flows_count", len(telemetry.get("findings", []))),
        "message": telemetry.get("message"),
        "capture_error": telemetry.get("capture_error"),
        "is_attack": finding0.get("is_attack", False) if finding0 else False,
        "attack_type": finding0.get("attack_type") if finding0 else None,
        "attack_probability": finding0.get("attack_probability") if finding0 else None,
        "confidence": finding0.get("confidence") if finding0 else None,
        "decision_threshold": finding0.get("decision_threshold") if finding0 else None,
        "passed_to_llama": telemetry.get("passed_to_llama", False),
        "output_type": telemetry.get("output_type", "no_output"),
        "output": telemetry.get("output"),
        "flow_info": {
            "flow_id": finding0.get("flow_id"),
            "src_ip": finding0.get("src_ip"),
            "src_port": finding0.get("src_port"),
            "dst_ip": finding0.get("dst_ip"),
            "dst_port": finding0.get("dst_port"),
            "protocol": finding0.get("protocol")
        } if finding0 else None,
        "shap_evidence": finding0.get("shap_evidence", []) if finding0 else [],
        "llama_prompt": telemetry.get("llama_prompt"),
        "llama_advisory": telemetry.get("llama_advisory"),
        "findings": telemetry["findings"],
        "latency_ms": finding0.get("latency_ms") if finding0 else None
    }
    return jsonify(response)


@app.route("/api/pipeline2", methods=["POST"])
def run_pipeline_2_endpoint():
    """
    Pipeline 2 Endpoint:
    User Input -> SMS Scam Detector -> Extract URL -> PhishGuard URL Detector -> Llama Prompt 2 -> Output
    """
    data = request.get_json(force=True) if request.data else {}
    user_input = data.get("user_input") or data.get("message") or ""
    enable_llm = data.get("enable_llm_reasoning", True)
    cross_check = data.get("cross_check_urls", True)

    pipe = get_pipeline()
    pipe.enable_llm_reasoning = bool(enable_llm)

    telemetry = pipe.run_pipeline_2(user_input, cross_check_urls=cross_check)

    sms_finding = next((f for f in telemetry["findings"] if f.get("module") == "SMS_Spam_Detector"), None)
    url_findings = [f for f in telemetry["findings"] if f.get("module") == "PhishGuard_URL_Detector"]
    extracted_urls = [f.get("url") for f in url_findings if f.get("url")]

    response = {
        "pipeline": "Pipeline 2: User Input -> SMS Scam Detector -> Extract URL -> PhishGuard URL Detector -> Llama Prompt 2 -> Output",
        "status": telemetry["status"],
        "timestamp": telemetry["timestamp"],
        "active_threats_count": telemetry["active_threats_count"],
        "user_input": user_input,
        "sms_telemetry": sms_finding,
        "extracted_urls": extracted_urls,
        "url_telemetry": url_findings,
        "llama_prompt": telemetry.get("llama_prompt"),
        "llama_advisory": telemetry.get("llama_advisory"),
        "findings": telemetry["findings"]
    }
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Bharat AI-SoC Edge Flask Server on http://0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

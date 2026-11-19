"""
Bharat AI-SoC Challenge - Problem Statement 3: On-Device Edge Security
Lightweight Flask + HTML5/CSS3/JS Web Interface & Telemetry API Server
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from unified_pipeline import (
    UnifiedThreatPipeline,
    extract_urls_from_text,
    normalize_sms_text,
    generate_llama_executorch_prompt
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
        "edge_models": {
            "sms_scam_detector": "ACTIVE (Stacking Ensemble, 25k dims, tau=0.390)",
            "phishguard_url": "ACTIVE (PyTorch MLP, 375k params)",
            "network_nids": "ACTIVE (13-Class XGBoost + SHAP, tau=0.3545)"
        },
        "downstream_engine": "Meta Llama 3.2 ExecuTorch 1B",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/api/analyze/sms", methods=["POST"])
def analyze_sms():
    data = request.get_json(force=True)
    message = data.get("message", "")
    cross_check = data.get("cross_check_urls", True)
    pipe = get_pipeline()
    telemetry = pipe.analyze_sms(message, cross_check_urls=cross_check)
    return jsonify(telemetry)


@app.route("/api/analyze/url", methods=["POST"])
def analyze_url():
    data = request.get_json(force=True)
    url = data.get("url", "")
    pipe = get_pipeline()
    telemetry = pipe.analyze_url(url)
    return jsonify(telemetry)


@app.route("/api/analyze/flow", methods=["POST"])
def analyze_flow():
    data = request.get_json(force=True)
    flow_row = data.get("flow", {})
    pipe = get_pipeline()
    df = pd.DataFrame([flow_row])
    telemetry = pipe.analyze_network_flow(df)
    return jsonify(telemetry)


@app.route("/api/analyze/unified", methods=["POST"])
def analyze_unified():
    data = request.get_json(force=True)
    sms_text = data.get("sms_text")
    url = data.get("url")
    flow_data = data.get("flow")

    flow_df = None
    if flow_data:
        flow_df = pd.DataFrame([flow_data])

    pipe = get_pipeline()
    telemetry = pipe.analyze_unified_event(sms_text=sms_text, url=url, flow_df=flow_df)
    return jsonify(telemetry)


@app.route("/api/sample/flows", methods=["GET"])
def sample_flows():
    sample_csv = WORKSPACE_ROOT / "data" / "sample_flows.csv"
    if not sample_csv.exists():
        sample_csv = WORKSPACE_ROOT / "nids" / "tests" / "fixtures" / "parity_rows.csv"
    if sample_csv.exists():
        df = pd.read_csv(sample_csv, nrows=15)
        records = df.to_dict(orient="records")
        return jsonify({"count": len(records), "flows": records})
    return jsonify({"count": 0, "flows": []})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Bharat AI-SoC Edge Flask Server on http://0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

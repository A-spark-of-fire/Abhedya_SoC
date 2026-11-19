# Bharat AI-SoC: Unified Multi-Model On-Device Threat Detection Pipeline

**Problem Statement 3 — On-Device Edge Security**
Target Hardware: **Linux / Raspberry Pi 5 (Bharat SoC)**

---

## Overview

This repository integrates three independent edge security modules into a **single unified Python backend pipeline** (`unified_pipeline.py`) and a **local web dashboard** (`app_dashboard.py`), enabling real-time, sub-millisecond threat detection entirely on-device — no cloud connectivity required.

| Module | Repository | Model | Accuracy | Latency |
|--------|-----------|-------|----------|---------|
| SMS Scam Detector | `A-spark-of-fire/SMS_Spam_Detector` | Stacking Ensemble (SVM+LR+NB) | 99.26% | < 1 ms |
| Phishing URL Detector | `SwastikSaha/PhishGuard` | PyTorch MLP (375k params) | — | < 2 ms |
| Network NIDS | `Abhil4sh777/nids` | 13-Class XGBoost + SHAP | — | < 2 ms |

---

## Repository Structure

```
Abhedya_SoC/
├── unified_pipeline.py          # Core pipeline: 3-module + aggregator + Llama prompt gen
├── app_dashboard.py             # Streamlit Web UI dashboard
├── app_flask.py                 # Lightweight Flask REST API server
├── templates/
│   └── index.html               # HTML5/CSS3/JS dashboard (served by Flask)
├── requirements.txt
├── data/
│   ├── sample_flows.csv         # 25-row curated NIDS test fixture (parity format)
│   └── sample_traffic.pcap      # Synthetic Ethernet/TCP capture for PCAP replay demo
├── SMS_Spam_Detector/           # Git submodule / cloned repo
│   ├── sms_classifier_model.pkl
│   ├── tfidf_word_vectorizer.pkl
│   ├── tfidf_char_vectorizer.pkl
│   ├── sms_decision_threshold.pkl
│   └── train_combined_sms.py
├── PhishGuard/                  # Git submodule / cloned repo
│   ├── phishguard_artifacts/
│   │   ├── phishguard_mlp.pth
│   │   ├── preprocessor.joblib
│   │   └── model_config.json
│   ├── live_classifier.py
│   └── mlp_model.py
└── nids/                        # Git submodule / cloned repo
    ├── models/
    │   ├── detector.joblib      # Binary XGBoost detector (tau=0.3545)
    │   └── classifier.joblib    # 13-class XGBoost classifier
    ├── features/contract.py
    ├── explain/shap_explainer.py
    └── pipeline.py
```

---

## Quick Start

### 1. Install Dependencies

```bash
# System packages (Linux / RPi5)
sudo apt update && sudo apt install -y git python3-pip

# Python packages
pip install -r requirements.txt

# For Raspberry Pi 5 (ARM64), install CPU-only PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Launch the Streamlit Dashboard

```bash
streamlit run app_dashboard.py --server.port 8501 --server.address 0.0.0.0
# Open browser at: http://localhost:8501  (or http://<rpi5-ip>:8501)
```

### 3. Launch the Flask REST API + HTML Dashboard

```bash
python app_flask.py
# API + HTML UI at: http://localhost:5000
```

### 4. Run CLI Pipeline Demo

```bash
python unified_pipeline.py --demo
```

### 5. Single-Shot CLI Inference

```bash
# Analyze a custom SMS
python unified_pipeline.py --sms "URGENT: Your SBI account locked. Update KYC: http://sbi-kyc.xyz"

# Analyze a URL
python unified_pipeline.py --url "http://sbi-banking-verify-update.xyz/login.php"

# Analyze network flow CSV
python unified_pipeline.py --flow-csv data/sample_flows.csv
```

---

## Module Integration Specifications

### Module 1: SMS Scam Detector

**De-Obfuscation Engine:**
- Spaced letter repair: `c l i c k` → `click`, `k y c` → `kyc`
- Dot obfuscations: `dr . ugs` → `drugs`, `gen . eric` → `generic`
- Leetspeak: `b@nk` → `bank`, `k-y-c` → `kyc`, `up1` → `upi`, `fr33` → `free`

**Feature Vector (25,049 dims):**
- 10,000 Word TF-IDF n-grams
- 15,000 Character TF-IDF n-grams
- 49 Structural/Domain signals (URL presence, phone number, currency symbols, 39 keyword flags)

**Telemetry Output:**
```json
{
  "module": "SMS_Spam_Detector",
  "is_threat": true,
  "threat_score": 0.9978,
  "threat_level": "HIGH",
  "decision_threshold": 0.390,
  "detected_triggers": ["kyc", "bank", "urgent", "update"],
  "latency_ms": 0.62
}
```

---

### Module 2: PhishGuard URL Detector

**Feature Extraction (1,334 dims after preprocessing):**
- 14 numeric features: URL length, Shannon entropy, slash count, query length, domain entropy, subdomain depth, digit count, special characters, hyphen count, HTTPS flag, IP flag, @ flag, keyword flag
- 1 categorical (TLD) → one-hot encoded

**Anomaly Checks:**
- Raw IP hostnames, `@` character in URL, multiple `//`
- High-risk TLDs: `.xyz`, `.top`, `.tk`, `.ml`, `.buzz`, etc.
- Sensitive auth keywords: `login`, `verify`, `kyc`, `banking`, `password`
- Typosquatting brands: `sbi`, `hdfc`, `paypal`, `google`, `microsoft`

**Telemetry Output:**
```json
{
  "module": "PhishGuard_URL_Detector",
  "is_phishing": true,
  "confidence_score": 1.0,
  "benign_score": 0.0,
  "domain_anomalies": ["High-risk TLD (.xyz)", "Excessive hyphens (3)"],
  "latency_ms": 0.85
}
```

---

### Module 3: Network NIDS

**Three Input Modes:**
| Mode | Input | Description |
|------|-------|-------------|
| 1 | Pre-extracted CSV | BCCC-format flow rows (121 columns) |
| 2 | PCAP file | Replayed via NTLFlowLyzer extractor |
| 3 | Live capture | `dumpcap` ring-buffer stream (Linux only) |

**13 Attack Classes:**
`DoS_Hulk`, `Port_Scan`, `DDoS_LOIT`, `FTP-Patator`, `DoS_GoldenEye`, `DoS_Slowhttptest`, `SSH-Patator`, `Botnet_ARES`, `DoS_Slowloris`, `Web_Brute_Force`, `Web_XSS`, `Web_SQL_Injection`, `Heartbleed`

**Two-Stage Inference:**
1. **Binary Detector** — XGBoost classifier at threshold `tau = 0.3545`
2. **Multi-class Classifier** — 13-class XGBoost on attack-flagged flows
3. **SHAP Explainer** — TreeExplainer top-5 feature attributions as mathematical evidence

**Telemetry Output:**
```json
{
  "module": "Network_NIDS",
  "flow_id": "parity-000",
  "is_attack": true,
  "attack_type": "SSH-Patator",
  "confidence": 0.4522,
  "attack_probability": 0.5359,
  "shap_evidence": [
    {"feature": "fwd_psh_flag_counts", "value": 497.42, "shap_value": 1.3097},
    {"feature": "fwd_packets_IAT_mean", "value": 232.37, "shap_value": -0.9651}
  ],
  "latency_ms": 1.14
}
```

---

## Cross-Module Automation

### Auto-Link Extractor
When an SMS is received, the pipeline **automatically extracts embedded URLs** (`http://`, `https://`, `bit.ly/`, `.xyz`, `.top`, shortlinks) and routes them to PhishGuard for simultaneous validation without any user intervention.

```python
from unified_pipeline import extract_urls_from_text
urls = extract_urls_from_text("URGENT: Update KYC at http://sbi-kyc.xyz immediately!")
# → ["http://sbi-kyc.xyz"]  (auto-forwarded to PhishGuard)
```

### Standardized Telemetry Aggregator

All module outputs are merged into a single JSON schema ready for downstream **Llama 3.2 ExecuTorch** prompt generation:

```json
{
  "status": "ALERT",
  "timestamp": "2026-11-19T18:30:00Z",
  "active_threats_count": 2,
  "findings": [
    {"module": "SMS_Spam_Detector", "is_threat": true, ...},
    {"module": "PhishGuard_URL_Detector", "is_phishing": true, ...},
    {"module": "Network_NIDS", "is_attack": true, "attack_type": "SSH-Patator", ...}
  ],
  "llama_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>..."
}
```

---

## Flask REST API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/health` | GET | Edge SoC health check |
| `POST /api/analyze/sms` | POST | SMS scam detection + auto URL extraction |
| `POST /api/analyze/url` | POST | PhishGuard URL analysis |
| `POST /api/analyze/flow` | POST | NIDS flow row classification + SHAP |
| `POST /api/analyze/unified` | POST | Full 3-module aggregated analysis |
| `GET /api/sample/flows` | GET | Retrieve sample NIDS flow fixtures |

---

## Downstream Llama 3.2 ExecuTorch Integration

The pipeline generates a fully-formatted **Llama 3.2 1B-Instruct chat prompt** in the ExecuTorch token format, ready to pass directly to the on-device runner:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the On-Device Security Copilot running on a low-power Edge SoC...
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
EDGE SECURITY TELEMETRY EVENT:
• Status: ALERT
• Active Threats Count: 2
...
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

Configure the ExecuTorch runner via environment variables:
```bash
export EXECUTORCH_PTE_PATH=/models_llm/xnnpack/Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte
export EXECUTORCH_TOKENIZER_PATH=/models_llm/tokenizer.json
export LLM_BACKEND=executorch
```

---

## Edge Deployment Notes

```bash
# Verify NIDS model and SHAP loading
python -c "from unified_pipeline import NetworkNIDS; n=NetworkNIDS(); print('NIDS OK')"

# Live capture on Linux (requires wireshark-common + user in wireshark group)
sudo apt install wireshark-common
sudo usermod -aG wireshark $USER
export NIDS_INTERFACE=eth0

# Run as a systemd service
cp nids/deploy/nids.service /etc/systemd/system/abhedya-soc.service
sudo systemctl enable --now abhedya-soc.service
```

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
├── .gitmodules                  # Git submodule configuration
├── modules/                     # Structured sub-system security repositories
│   ├── __init__.py              # Python module wiring & sys.path configuration
│   ├── sms_spam_detector/       # Submodule: A-spark-of-fire/SMS_Spam_Detector
│   │   ├── sms_classifier_model.pkl
│   │   ├── tfidf_word_vectorizer.pkl
│   │   ├── tfidf_char_vectorizer.pkl
│   │   ├── sms_decision_threshold.pkl
│   │   └── train_combined_sms.py
│   ├── phishguard/              # Submodule: SwastikSaha/PhishGuard
│   │   ├── phishguard_artifacts/
│   │   │   ├── phishguard_mlp.pth
│   │   │   ├── preprocessor.joblib
│   │   │   └── model_config.json
│   │   ├── live_classifier.py
│   │   └── mlp_model.py
│   └── nids/                    # Submodule: Abhil4sh777/nids
│       ├── models/
│       │   ├── detector.joblib  # Binary XGBoost detector (tau=0.3545)
│       │   └── classifier.joblib# 13-class XGBoost classifier
│       ├── features/contract.py
│       ├── explain/shap_explainer.py
│       └── pipeline.py
├── scripts/
│   ├── download_llama.sh        # Bash model asset downloader from Hugging Face
│   └── download_llama.py        # Cross-platform Python downloader for PTE & tokenizer
├── models/
│   └── llama/                   # Local ExecuTorch Llama 3.2 1B-Instruct model assets
├── llama_prompt_builder.py      # Llama 3.2 Instruct prompt constructor & grounded fallback
├── executorch_runner.py         # ExecuTorch 1.5.1 C++ CLI runner wrapper & safety fallback
├── unified_pipeline.py          # Core pipeline: 3-module + aggregator + ExecuTorch stitching
├── app_dashboard.py             # Streamlit Web UI dashboard with dedicated AI Reasoning Card
├── app_flask.py                 # Lightweight Flask REST API server with LLM reasoning toggle
├── templates/
│   └── index.html               # Cyber-styled HTML5/CSS3/JS dashboard (served by Flask)
├── tests/
│   └── test_llama_stitch.py     # End-to-end integration test for Llama reasoning & telemetry
├── requirements.txt             # Merged unified dependencies across all 3 modules
└── data/
    ├── sample_flows.csv         # 25-row curated NIDS test fixture (parity format)
    └── sample_traffic.pcap      # Synthetic Ethernet/TCP capture for PCAP replay demo
```

---

## Quick Start

### 1. Clone & Initialize Submodules

```bash
# Clone with submodules
git clone --recursive <repo-url>
cd Abhedya_SoC

# Or if repository is already cloned:
git submodule update --init --recursive
```

### 2. Install Dependencies

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

## Local ExecuTorch Llama 3.2 Reasoning Engine Integration

The pipeline embeds local, on-device generative AI reasoning using **Meta Llama 3.2 (1B-Instruct 4-bit PTE)** compiled for ExecuTorch 1.5.1 Arm64 XNNPACK. When threat alerts occur, raw technical telemetry from all active edge sensors is synthesized into a plain-English 3-part advisory for non-technical users.

### 1. Download Model Assets

Download the pre-quantized ExecuTorch 4-bit PTE model (~1.0 GB RAM footprint) and tokenizer:

```bash
# Using bash script (Linux / Raspberry Pi 5):
chmod +x scripts/download_llama.sh
./scripts/download_llama.sh

# Or using the cross-platform Python downloader:
python scripts/download_llama.py

# Download tokenizer only (for testing without full 1.1GB weights):
python scripts/download_llama.py --tokenizer-only
```

Model assets are stored at `models/llama/` (or `$HOME/models_llm/`):
- `Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte` (4-bit XNNPACK quantized)
- `tokenizer.json` (Hugging Face / ExecuTorch tokenizer configuration)
- `xnnpack/config.json`

Target Hugging Face Repository: `experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch`

### 2. Zero-Shot Prompt Construction (`llama_prompt_builder.py`)

`llama_prompt_builder.py` translates raw telemetry into a zero-shot chat prompt using strict Llama 3.2 special tokens:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are the on-device security assistant for Abhedya SoC / Trinetra AI.
Your job is to translate technical threat telemetry from 3 on-device ML sensors (SMS Scam Detector, PhishGuard URL Analyzer, and Network NIDS) into a simple, plain-English advisory for non-technical users.

STRICT RULES:
1. Ground your answer strictly in the provided JSON telemetry. Never invent threats not present in the payload.
2. Output a 3-part structured alert:
   - ⚠️ **Threat Summary:** What was detected in plain English.
   - 🔍 **Technical Reason:** Why it is dangerous (explain SHAP metrics, suspicious TLDs, or SMS trigger words simply).
   - 🛡️ **Recommended Actions:** Exactly 3 clear safety steps the user should take.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Review the following on-device threat telemetry and generate the plain-language safety alert:

```json
{TELEMETRY_JSON}
```
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### 3. ExecuTorch Engine Runner & Graceful Fallback (`executorch_runner.py`)

- **Class:** `ExecuTorchLlamaEngine`
- **Inference Parameters:** `temperature=0.2`, `max_new_tokens=256`, 15-second subprocess timeout.
- **Binary Detection:** Automatically scans PATH, `EXECUTORCH_RUNNER_BIN`, and `$HOME/executorch/` for `llama_main` or `executorch_llama_runner`.
- **Deterministic Grounded Fallback:** If the C++ runner binary or `.pte` weights are absent (e.g. dev workstations or before compilation on Pi 5), the engine executes `generate_grounded_fallback_advisory()`. This guarantees that dashboards and automated scripts never crash and always receive the exact 3-part structured advisory grounded in telemetry findings.

### 4. Running the Verification Smoke Test

Execute the automated test suite simulating an SMS smishing attack with an embedded phishing link (`http://sbi-kyc.top`) plus a concurrent NIDS DoS attack flow:

```bash
python -m unittest tests/test_llama_stitch.py
```

All 6 test cases verify:
- Llama 3.2 Instruct token prompt formatting
- Deterministic grounded fallback alert formatting (exact 3 sections, exactly 3 action steps)
- Multi-vector correlation alert generation & prompt/advisory stitching
- Clean traffic baseline handling
- UI toggle disablement behavior
- ExecuTorch engine fallback safety without binaries

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

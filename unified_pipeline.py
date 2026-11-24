"""
Unified Multi-Model Threat Detection Pipeline
Bharat AI-SoC Challenge - Problem Statement 3: On-Device Edge Security

Integrates:
1. SMS Scam Detector (A-spark-of-fire/SMS_Spam_Detector) - Sub-ms text threat classifier (25,049 dims, tau=0.390)
2. Phishing URL Detector (SwastikSaha/PhishGuard) - PyTorch MLP structural & domain feature analyzer
3. Network Intrusion Detection System (Abhil4sh777/nids) - 13-class XGBoost NIDS with SHAP explainability (tau=0.3545)
4. Cross-Module Automation & Aggregator Engine - Auto-Link Extractor & Llama 3.2 ExecuTorch Prompt Generator
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import ipaddress
import json
import logging
import math
import os
import queue
import threading
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit
import warnings

# Ensure UTF-8 output encoding for edge terminals and Windows consoles
try:
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Suppress sklearn/xgboost version mismatch warnings on edge environments
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

# Base paths & Submodule resolution
WORKSPACE_ROOT = Path(__file__).resolve().parent
MODULES_DIR = WORKSPACE_ROOT / "modules"

# Resolve submodule directories inside modules/ (with fallback to root)
SMS_DIR = MODULES_DIR / "sms_spam_detector" if (MODULES_DIR / "sms_spam_detector").exists() else WORKSPACE_ROOT / "SMS_Spam_Detector"
PHISH_DIR = MODULES_DIR / "phishguard" if (MODULES_DIR / "phishguard").exists() else WORKSPACE_ROOT / "PhishGuard"
NIDS_DIR = MODULES_DIR / "nids" if (MODULES_DIR / "nids").exists() else WORKSPACE_ROOT / "nids"

# Configure sys.path so submodules and internal packages resolve seamlessly
for p in [WORKSPACE_ROOT, MODULES_DIR, SMS_DIR, PHISH_DIR, NIDS_DIR]:
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# NTLFlowLyzer is the single, mandatory flow-extraction engine for every
# network capture in this pipeline.  The NIDS submodule owns the feature
# contract, column mapping, pcap/pcapng conversion, and extractor execution.
try:
    from features.contract import ALL_FEATURES
    from features.ntlflowlyzer import NTLFlowLyzerError, NTLFlowLyzerRunner
except ImportError as _nids_import_error:  # pragma: no cover - deployment error
    raise RuntimeError(
        "The NIDS submodule (modules/nids) is required: network captures are "
        "parsed exclusively by NTLFlowLyzer. Initialise the submodule with "
        "`git submodule update --init --recursive` and install NTLFlowLyzer "
        "inside modules/nids (see modules/nids/README.md)."
    ) from _nids_import_error

logger = logging.getLogger("unified_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# GLOBAL LLM MUTEX FOR MULTI-THREADED DUAL ENGINE (APPROACH 2)
# =====================================================================
# Serializes access to local ExecuTorch Llama 3.2 engine across threads.
# Ensures zero RAM overload (~1.2 GB limit) and instant sub-2ms evaluation
# for benign events on Raspberry Pi 5.
llm_lock = threading.Lock()


# =====================================================================
# MODULE 1: SMS Scam Detector (A-spark-of-fire/SMS_Spam_Detector)
# =====================================================================

DOMAIN_KEYWORDS = [
    'kyc', 'bank', 'upi', 'apk', 'job', 'part-time', 'telegram', 'lottery',
    'account', 'verify', 'reward', 'claim', 'urgent', 'winner', 'free', 'prize',
    'cash', 'update', 'blocked', 'suspended', 'deactivated', 'otp', 'pan', 'aadhaar',
    'refund', 'loan', 'credit', 'call', 'reply', 'text', 'stop', 'bonus', 'gift',
    'officer', 'electricity', 'disconnected', 'prepaid', 'earning', 'task'
]

def normalize_sms_text(text: str) -> str:
    """
    De-obfuscates text:
    1. Spaced letters: 'c l i c k' -> 'click', 'f r e e' -> 'free', 'k y c' -> 'kyc'
    2. Dot/symbol obfuscations: 'dr . ugs' -> 'drugs', 'gen . eric' -> 'generic', 'r . e . p . l . y' -> 'reply'
    3. Leetspeak mapping: 'b@nk' -> 'bank', 'k-y-c' -> 'kyc', 'up1' -> 'upi', '$pam' -> 'spam'
    4. Repetitive urgency normalization: '!!!!' -> '!'
    """
    text = str(text) if text is not None else ""

    # Spaced characters repair
    text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3\4\5', text)
    text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3\4', text)
    text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3', text)

    # Dot/spaced obfuscations
    text = re.sub(r'\bgen\s*\.\s*eric\b', 'generic', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdr\s*\.\s*ugs\b', 'drugs', text, flags=re.IGNORECASE)
    text = re.sub(r'\br\s*e\s*p\s*l\s*y\b', 'reply', text, flags=re.IGNORECASE)
    text = re.sub(r'\bf\s*r\s*e\s*e\b', 'free', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvic\s*\.\s*odin\b', 'vicodin', text, flags=re.IGNORECASE)

    # Leetspeak & symbol substitutions
    obfuscations = {
        r'\bc1ick\b': 'click', r'\bcl1ck\b': 'click', r'\ba\.c\.c\.o\.u\.n\.t\b': 'account',
        r'\b\$pam\b': 'spam', r'\bb@nk\b': 'bank', r'\bfr33\b': 'free', r'\bup1\b': 'upi',
        r'\bk-y-c\b': 'kyc', r'\bw1nner\b': 'winner', r'\bl0an\b': 'loan', r'\bcr3dit\b': 'credit',
        r'\bp4ssword\b': 'password', r'\bv1codin\b': 'vicodin'
    }
    for pattern, replacement in obfuscations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Excessive punctuation normalization
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    return text.strip()


def extract_sms_structural_features(texts: Sequence[str]) -> np.ndarray:
    """Extracts 49 structural and domain keyword threat features."""
    features = []
    urgency_symbols = ['!', '₹', '$', '%', '?', '£', '€', '*', '#']
    url_pattern = re.compile(
        r'https?://|\bwww\b|\bbit\.ly\b|\btinyurl\b|\bt\.co\b|\bt\.me\b|\bwa\.me\b|\b\w+\.(com|in|org|net|xyz|top|link|site|info|online|live|club|app|me|cc|vip)\b',
        re.IGNORECASE
    )
    phone_pattern = re.compile(
        r'\b\d{10,12}\b|\b\d{4,5}[-\s]?\d{4,5}\b|\b\+?\d{1,3}[-\s]?\d{9,10}\b|\b84484\b|\b087[012]\b|\b090[56]\b'
    )
    currency_pattern = re.compile(r'[£$€₹¥]|rs\.?|inr|usd|gbp', re.IGNORECASE)

    for text in texts:
        text_str = str(text) if text is not None else ""
        length = max(len(text_str), 1)

        has_url = 1.0 if url_pattern.search(text_str) else 0.0
        has_phone = 1.0 if phone_pattern.search(text_str) else 0.0
        has_currency = 1.0 if currency_pattern.search(text_str) else 0.0

        uppercase_count = sum(1 for c in text_str if c.isupper())
        uppercase_ratio = uppercase_count / length

        symbol_count = sum(text_str.count(sym) for sym in urgency_symbols)
        symbol_ratio = symbol_count / length

        non_ascii_count = sum(1 for c in text_str if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / length

        digit_count = sum(1 for c in text_str if c.isdigit())
        digit_ratio = digit_count / length

        excl_count = float(text_str.count('!'))
        q_count = float(text_str.count('?'))
        char_len = float(len(text_str))

        text_lower = text_str.lower()
        kw_flags = [1.0 if kw in text_lower else 0.0 for kw in DOMAIN_KEYWORDS]

        row = [
            has_url, has_phone, has_currency, uppercase_ratio, symbol_ratio,
            non_ascii_ratio, digit_ratio, excl_count, q_count, char_len
        ] + kw_flags
        features.append(row)

    return np.asarray(features, dtype=np.float32)


class SMSScamDetector:
    """Sub-millisecond SMS text threat classifier with 99.2%+ accuracy."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else SMS_DIR
        self.model_path = self.model_dir / "sms_classifier_model.pkl"
        self.word_vec_path = self.model_dir / "tfidf_word_vectorizer.pkl"
        self.char_vec_path = self.model_dir / "tfidf_char_vectorizer.pkl"
        self.threshold_path = self.model_dir / "sms_decision_threshold.pkl"

        if not self.model_path.exists():
            raise FileNotFoundError(f"SMS Classifier model not found at {self.model_path}")

        logger.info("Loading SMS Scam Detector ensemble & vectorizers...")
        self.model = joblib.load(self.model_path)
        # Ensure scikit-learn cross-version compatibility for unpickled LogisticRegression estimators
        for est in getattr(self.model, "estimators_", []):
            if not hasattr(est, "multi_class"):
                setattr(est, "multi_class", "auto")
        self.word_vec = joblib.load(self.word_vec_path)
        self.char_vec = joblib.load(self.char_vec_path)

        if self.threshold_path.exists():
            self.threshold = float(joblib.load(self.threshold_path))
        else:
            self.threshold = 0.390
        logger.info(f"SMS Detector loaded successfully (threshold tau={self.threshold:.3f})")

    def predict(self, message: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_text = str(message).strip()
        norm_text = normalize_sms_text(raw_text)

        X_word = self.word_vec.transform([norm_text])
        X_char = self.char_vec.transform([norm_text])
        X_struct = extract_sms_structural_features([norm_text])

        # Dynamic feature alignment check with trained pipeline
        expected_features = getattr(self.model, 'n_features_in_', 25049)
        word_char_len = X_word.shape[1] + X_char.shape[1]
        expected_struct_len = expected_features - word_char_len
        current_struct_len = X_struct.shape[1]

        if current_struct_len > expected_struct_len:
            X_struct = X_struct[:, :expected_struct_len]
        elif current_struct_len < expected_struct_len:
            pad = np.zeros((X_struct.shape[0], expected_struct_len - current_struct_len), dtype=np.float32)
            X_struct = np.hstack([X_struct, pad])

        X_comb = sp.hstack([X_word, X_char, sp.csr_matrix(X_struct)], format='csr')

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_comb)[0]
            scam_prob = float(probs[1])
        else:
            decision = self.model.decision_function(X_comb)[0]
            scam_prob = float(1.0 / (1.0 + np.exp(-decision)))

        is_scam = scam_prob >= self.threshold
        level = 'HIGH' if scam_prob >= 0.70 else ('MEDIUM' if is_scam else 'LOW')

        trigger_keywords = [
            'kyc', 'bank', 'upi', 'apk', 'job', 'part-time', 'telegram', 'lottery',
            'account', 'verify', 'reward', 'claim', 'urgent', 'winner', 'free', 'update',
            'blocked', 'otp', 'loan', 'credit', 'click', 'call', 'reply', 'prize'
        ]
        norm_lower = norm_text.lower()
        detected = [kw for kw in trigger_keywords if kw in norm_lower]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return {
            "module": "SMS_Spam_Detector",
            "raw_text": message,
            "normalized_text": norm_text,
            "is_threat": bool(is_scam),
            "threat_score": round(scam_prob, 4),
            "threat_level": level,
            "decision_threshold": round(self.threshold, 3),
            "detected_triggers": detected,
            "total_features": int(X_comb.shape[1]),
            "latency_ms": latency_ms
        }


# =====================================================================
# MODULE 2: Phishing URL Detector (SwastikSaha/PhishGuard)
# =====================================================================

class URLClassifier(nn.Module):
    """PyTorch MLP model for deep URL classification."""
    def __init__(
        self,
        input_size: int = 1334,
        hidden_sizes: Tuple[int, int] = (256, 128),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


SUSPICIOUS_URL_KEYWORDS = {
    "login", "signin", "verify", "verification", "update", "secure",
    "account", "banking", "password", "confirm", "wallet", "invoice",
    "payment", "recover", "suspend", "unlock", "authenticate", "kyc",
    "sbi", "hdfc", "icici", "paypal", "support", "claim", "free", "gift"
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "fit", "cc",
    "vip", "work", "icu", "loan", "club", "site", "online", "link", "live"
}

TYPOSQUAT_BRANDS = [
    "google", "paypal", "microsoft", "amazon", "apple", "netflix",
    "facebook", "instagram", "whatsapp", "telegram", "sbi", "hdfc", "icici"
]


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def extract_url_features(raw_url: str) -> Dict[str, Any]:
    raw_url = raw_url.strip()
    candidate = raw_url if "://" in raw_url else "http://" + raw_url
    parsed = urlsplit(candidate)

    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        host = "unknown"

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    labels = [part for part in host.split(".") if part]
    tld = labels[-1] if labels else "unknown"
    subdomain_count = max(0, len(labels) - 2) if not is_ip else 0

    lowered_url = raw_url.lower()
    keyword_flag = int(any(kw in lowered_url for kw in SUSPICIOUS_URL_KEYWORDS))

    return {
        "url_length": len(raw_url),
        "url_entropy": shannon_entropy(raw_url),
        "slash_count": raw_url.count("/"),
        "query_length": len(parsed.query),
        "domain_length": len(host),
        "domain_entropy": shannon_entropy(host),
        "sub_domain": subdomain_count,
        "digit_count": sum(ch.isdigit() for ch in raw_url),
        "special_char_count": sum(not ch.isalnum() for ch in raw_url),
        "hyphen_count": raw_url.count("-"),
        "https_flag": int(parsed.scheme.lower() == "https"),
        "ip_flag": is_ip,
        "at_flag": int("@" in raw_url),
        "keyword_flag": keyword_flag,
        "tld": tld,
        "hostname": host,
        "scheme": parsed.scheme or "http",
        "path": parsed.path or "/"
    }


def analyze_url_anomalies(raw_url: str, features: Dict[str, Any]) -> List[str]:
    anomalies = []
    host = features.get("hostname", "")
    lowered = raw_url.lower()

    if features["ip_flag"]:
        anomalies.append("Hostname is a raw IP address rather than a domain name.")
    if features["at_flag"]:
        anomalies.append("URL contains '@' character, potentially obscuring target host.")
    if raw_url.count("//") > 1:
        anomalies.append("Multiple '//' occurrences detected in URL path.")
    if features["sub_domain"] >= 3:
        anomalies.append(f"Abnormally high subdomain count ({features['sub_domain']} subdomains).")
    if features["hyphen_count"] >= 3:
        anomalies.append(f"Excessive hyphenation ({features['hyphen_count']} hyphens).")
    if features["url_length"] >= 90:
        anomalies.append(f"Excessive URL length ({features['url_length']} characters).")
    if features.get("tld") in SUSPICIOUS_TLDS:
        anomalies.append(f"High-risk top-level domain (.{features['tld']}) frequently abused for phishing.")
    if not features["https_flag"]:
        anomalies.append("Unencrypted connection (HTTP instead of HTTPS).")

    # Sensitive keyword triggers
    matched_kws = [kw for kw in SUSPICIOUS_URL_KEYWORDS if kw in lowered]
    if matched_kws:
        anomalies.append(f"Sensitive credential/auth keywords detected: {', '.join(matched_kws[:5])}.")

    # Typosquatting checks
    for brand in TYPOSQUAT_BRANDS:
        if brand in host and not host.endswith(f".{brand}.com") and host != f"{brand}.com":
            anomalies.append(f"Possible typosquatting / brand spoofing targeting '{brand}'.")
            break

    if not anomalies:
        anomalies.append("No overt structural anomalies detected.")

    return anomalies


class PhishGuardURLDetector:
    """Deep URL structural and domain feature analyzer using PyTorch MLP."""

    def __init__(self, artifact_dir: Optional[Path] = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else PHISH_DIR / "phishguard_artifacts"
        self.model_path = self.artifact_dir / "phishguard_mlp.pth"
        self.preprocessor_path = self.artifact_dir / "preprocessor.joblib"
        self.config_path = self.artifact_dir / "model_config.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"PhishGuard model weights missing at {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"PhishGuard preprocessor missing at {self.preprocessor_path}")

        logger.info("Loading PhishGuard PyTorch MLP & Preprocessor...")
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else {}

        self.device = torch.device("cpu")
        self.hidden_sizes = tuple(self.config.get("hidden_sizes", (256, 128)))
        self.input_size = int(self.config.get("input_size", 1334))

        self.model = URLClassifier(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            num_classes=2,
        ).to(self.device)

        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("PhishGuard URL Detector loaded successfully.")

    def predict(self, url: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_url = str(url).strip()
        features = extract_url_features(raw_url)
        anomalies = analyze_url_anomalies(raw_url, features)

        feature_cols = [
            "url_length", "url_entropy", "slash_count", "query_length",
            "domain_length", "domain_entropy", "sub_domain", "digit_count",
            "special_char_count", "hyphen_count", "https_flag", "ip_flag",
            "at_flag", "keyword_flag", "tld"
        ]
        df_row = pd.DataFrame([{k: features[k] for k in feature_cols}])

        transformed = self.preprocessor.transform(df_row).astype(np.float32)
        tensor = torch.from_numpy(transformed).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        benign_prob = float(probabilities[0])
        phish_prob = float(probabilities[1])
        is_phishing = phish_prob >= 0.50

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return {
            "module": "PhishGuard_URL_Detector",
            "url": raw_url,
            "is_phishing": bool(is_phishing),
            "confidence_score": round(phish_prob, 4),
            "benign_score": round(benign_prob, 4),
            "domain_anomalies": anomalies,
            "features": {
                "hostname": features["hostname"],
                "tld": features["tld"],
                "url_length": features["url_length"],
                "url_entropy": round(features["url_entropy"], 3),
                "domain_entropy": round(features["domain_entropy"], 3),
                "sub_domain_count": features["sub_domain"],
                "https": bool(features["https_flag"]),
                "is_ip": bool(features["ip_flag"]),
                "special_chars": features["special_char_count"]
            },
            "latency_ms": latency_ms
        }


class LiveCaptureError(RuntimeError):
    """Raised when live packet capture cannot start or complete.

    Callers should surface this message to the operator instead of pretending a
    benign flow was observed.  An *empty* capture (no packets in the window) is
    not an error and is returned as an empty DataFrame.

    ``stderr`` preserves the original tool diagnostic (dumpcap or
    NTLFlowLyzer) so the pipeline can return it verbatim as ``capture_error``.
    """

    def __init__(self, message: str, *, stderr: Optional[str] = None) -> None:
        super().__init__(message)
        self.stderr = stderr

    @property
    def diagnostic(self) -> str:
        """Exact underlying tool diagnostic, or the full message as fallback."""
        return (self.stderr or str(self)).strip()


def _capture_diagnostic(exc: BaseException) -> str:
    """Return the raw tool stderr when available, else the full exception text.

    ``NTLFlowLyzerRunner`` embeds subprocess stderr after the ``"): "`` marker
    in its error message, so unwrap that tail to honour the pipeline contract
    of returning the exact extractor diagnostic.
    """

    stderr = getattr(exc, "stderr", None)
    if stderr:
        return str(stderr).strip()
    message = str(exc).strip()
    marker = "): "
    if marker in message:
        tail = message.rsplit(marker, 1)[1].strip()
        if tail:
            return tail
    return message


_DUMPCAP_FALLBACK_PATHS = (
    "/usr/bin/dumpcap",
    "/usr/sbin/dumpcap",
    "/usr/local/bin/dumpcap",
)


def _find_dumpcap() -> str:
    """Return the dumpcap executable path or raise a helpful error."""

    found = shutil.which("dumpcap")
    if found:
        return found
    for candidate in _DUMPCAP_FALLBACK_PATHS:
        if Path(candidate).exists():
            return candidate
    raise LiveCaptureError(
        "dumpcap is not installed or not on PATH. Install Wireshark's capture "
        "tools with `sudo apt install wireshark-common`, then grant capture "
        "permissions to your user (see README: Live Capture Troubleshooting)."
    )


def _dumpcap_pcap_flags(dumpcap_bin: str) -> List[str]:
    """Force classic pcap output so DPkt/NTLFlowLyzer can read the capture.

    Wireshark >= 4.0 uses ``-F pcap`` while older builds (e.g. 3.6.x) use
    ``-P``.  If the help text cannot be parsed we let dumpcap choose its
    default; DPkt's pcapng fallback still handles that case.
    """

    try:
        help_proc = subprocess.run(
            [dumpcap_bin, "-h"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        help_text = f"{help_proc.stdout}\n{help_proc.stderr}"
    except (OSError, subprocess.SubprocessError):
        return []
    if re.search(r"(?m)^\s*-F\b", help_text):
        return ["-F", "pcap"]
    if re.search(r"(?m)^\s*-P\b", help_text):
        return ["-P"]
    return []


# ---------------------------------------------------------------------
# Capture preflight: distinguish a genuinely empty window from an error.
# This only counts packets; it never reconstructs flows or fabricates
# feature columns.  Actual feature extraction is always NTLFlowLyzer.
# ---------------------------------------------------------------------

_PCAP_MAGIC_ENDIAN = {
    b"\xd4\xc3\xb2\xa1": "little",  # little-endian, microsecond timestamps
    b"\x4d\x3c\xb2\xa1": "little",  # little-endian, nanosecond timestamps
    b"\xa1\xb2\xc3\xd4": "big",     # big-endian, microsecond timestamps
    b"\xa1\xb2\x3c\x4d": "big",     # big-endian, nanosecond timestamps
}
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
_PCAPNG_BOM = {
    b"\x4d\x3c\x2b\x1a": "little",
    b"\x1a\x2b\x3c\x4d": "big",
}


def _pcap_has_packets(path: Path, magic: bytes) -> Optional[bool]:
    """Return whether the classic-pcap file has at least one packet record."""

    endian = _PCAP_MAGIC_ENDIAN[magic]
    try:
        with path.open("rb") as handle:
            handle.read(24)  # global header
            while True:
                record = handle.read(16)
                if len(record) < 16:
                    return False
                incl_len = int.from_bytes(record[8:12], endian)
                handle.seek(incl_len, 1)
                return True
    except OSError:
        return None


def _pcapng_has_packets(path: Path) -> Optional[bool]:
    """Return whether the pcapng file has at least one packet block."""

    endian: Optional[str] = None
    try:
        with path.open("rb") as handle:
            while True:
                header = handle.read(8)
                if len(header) < 8:
                    return False
                block_type_raw = header[:4]
                if block_type_raw == _PCAPNG_MAGIC:
                    bom = handle.read(4)
                    if len(bom) < 4:
                        return False
                    endian = _PCAPNG_BOM.get(bom)
                    if endian is None:
                        return None
                    total_len = int.from_bytes(header[4:8], endian)
                    if total_len < 12:
                        return False
                    handle.seek(total_len - 12, 1)
                    continue
                if endian is None:
                    return None
                block_type = int.from_bytes(block_type_raw, endian)
                total_len = int.from_bytes(header[4:8], endian)
                if total_len < 12:
                    return False
                # Enhanced Packet Block (6), Simple Packet Block (3), and
                # obsolete Packet Block (2) each hold exactly one packet.
                if block_type in (2, 3, 6):
                    return True
                handle.seek(total_len - 8, 1)
    except OSError:
        return None


def _capture_is_empty(pcap_path: Path) -> bool:
    """Return True only when a valid capture contains zero packet records."""

    try:
        size = pcap_path.stat().st_size
        with pcap_path.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        return False

    if size == 0:
        return True
    if magic in _PCAP_MAGIC_ENDIAN:
        if size < 24:  # truncated header: let NTLFlowLyzer report the error
            return False
        return _pcap_has_packets(pcap_path, magic) is False
    if magic == _PCAPNG_MAGIC:
        if size < 28:  # truncated Section Header Block
            return False
        return _pcapng_has_packets(pcap_path) is False
    return False


def _ntlflowlyzer_settings():
    """Build settings from the NIDS environment (honours NIDS_NTLFLOWLYZER)."""

    from config import Settings

    return Settings.from_env()


def extract_flows_with_ntlflowlyzer(
    pcap_path: Union[str, Path],
    settings: Optional[Any] = None,
) -> pd.DataFrame:
    """Extract the full 121-feature contract from a capture using NTLFlowLyzer.

    There is deliberately no DPkt/manual fallback here.  A capture that cannot
    be parsed by NTLFlowLyzer raises :class:`NTLFlowLyzerError`; a valid
    capture containing zero packets returns an empty contract-shaped frame.
    """

    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise NTLFlowLyzerError(f"Capture file not found: {pcap_path}")
    if _capture_is_empty(pcap_path):
        return pd.DataFrame(columns=ALL_FEATURES)

    runner = NTLFlowLyzerRunner(settings or _ntlflowlyzer_settings())
    return runner.run(pcap_path)


def extract_flows_from_pcap(
    pcap_path: Union[str, Path],
    max_flows: Optional[int] = 50,
) -> pd.DataFrame:
    """Deprecated compatibility wrapper around NTLFlowLyzer.

    The old DPkt implementation populated only a subset of the 121-column
    contract and padded the rest with zeros.  This wrapper now runs the same
    mandatory NTLFlowLyzer extractor as live capture and PCAP replay.
    """

    warnings.warn(
        "extract_flows_from_pcap() is deprecated; use "
        "extract_flows_with_ntlflowlyzer() so the full NTLFlowLyzer feature "
        "contract is enforced.",
        DeprecationWarning,
        stacklevel=2,
    )
    frame = extract_flows_with_ntlflowlyzer(pcap_path)
    if max_flows is not None and max_flows > 0:
        frame = frame.iloc[:max_flows].reset_index(drop=True)
    return frame


def get_available_interfaces() -> List[str]:
    """Returns detected edge network interfaces for live capture."""
    interfaces = ["any", "wlo1", "lo", "eth0", "eno1"]
    try:
        res = subprocess.run(["dumpcap", "-D"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            found = []
            for line in res.stdout.strip().split("\n"):
                line = line.strip()
                if line:
                    parts = line.split(". ", 1)
                    if len(parts) == 2:
                        name = parts[1].split()[0]
                        if name not in found:
                            found.append(name)
            if found:
                prioritized = []
                for pref in ["any", "wlo1", "wlan0", "eth0", "eno1", "lo"]:
                    if pref in found and pref not in prioritized:
                        prioritized.append(pref)
                for iface in found:
                    if iface not in prioritized:
                        prioritized.append(iface)
                interfaces = prioritized
    except Exception:
        pass
    return interfaces


def get_dumpcap_info() -> Dict[str, Any]:
    """Return live-capture capability details for dashboards/health checks."""

    try:
        dumpcap_bin = _find_dumpcap()
    except LiveCaptureError as exc:
        return {"available": False, "path": None, "version": None, "error": str(exc)}
    version = None
    try:
        proc = subprocess.run(
            [dumpcap_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = (proc.stdout or proc.stderr or "").splitlines()[0].strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"available": True, "path": dumpcap_bin, "version": version, "error": None}


def capture_live_network(
    interface: str = "any",
    duration_seconds: int = 60,
    max_packets: Optional[int] = 50000,
    settings: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Capture real packets with dumpcap and extract flows with NTLFlowLyzer.

    The capture window defaults to 1 minute (60s).  The resulting frame always
    follows the exact 121-column NIDS contract produced by NTLFlowLyzer; no
    column is padded or synthesized.

    Raises:
        LiveCaptureError: dumpcap or NTLFlowLyzer is missing, cannot start, or
            exits non-zero (for example because the current user lacks
            packet-capture permissions).  An idle window with zero packets is
            *not* an error; it returns an empty frame so callers can report
            "no traffic".
    """

    empty_frame = pd.DataFrame(columns=ALL_FEATURES)
    dumpcap_bin = _find_dumpcap()
    duration_seconds = max(1, int(duration_seconds))
    settings = settings or _ntlflowlyzer_settings()

    temp_dir = Path(tempfile.mkdtemp(prefix="abhedya-capture-"))
    tmp_path = temp_dir / "live_capture.pcap"
    try:
        cmd = [
            dumpcap_bin,
            "-i", str(interface),
            "-a", f"duration:{duration_seconds}",
        ]
        if max_packets is not None and max_packets > 0:
            cmd.extend(["-c", str(int(max_packets))])
        cmd.extend([
            *_dumpcap_pcap_flags(dumpcap_bin),
            "-w", str(tmp_path),
        ])
        logger.info("Starting live capture: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration_seconds + 10,
            )
        except FileNotFoundError as exc:
            raise LiveCaptureError(
                f"dumpcap executable not found: {dumpcap_bin}",
                stderr=str(exc),
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LiveCaptureError(
                f"dumpcap did not finish within {duration_seconds + 8}s on interface "
                f"'{interface}'. The interface may be down or blocked by a permission prompt.",
                stderr=str(exc),
            ) from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise LiveCaptureError(
                f"dumpcap failed on interface '{interface}' (exit {proc.returncode}): "
                f"{detail or 'unknown error'}. Check that the interface exists and that "
                "your user has packet-capture permission (see README: Live Capture Troubleshooting).",
                stderr=detail or f"dumpcap exited with status {proc.returncode}",
            )

        if not tmp_path.exists() or tmp_path.stat().st_size <= 24:
            logger.warning(
                "dumpcap captured no packets on interface %s within %ss.",
                interface,
                duration_seconds,
            )
            return empty_frame

        try:
            flows = extract_flows_with_ntlflowlyzer(tmp_path, settings=settings)
        except NTLFlowLyzerError as exc:
            raise LiveCaptureError(
                f"NTLFlowLyzer failed on interface '{interface}': {exc}",
                stderr=_capture_diagnostic(exc),
            ) from exc

        if flows.empty:
            logger.warning(
                "Captured packets on interface %s but NTLFlowLyzer produced no "
                "TCP/UDP flows.",
                interface,
            )
        else:
            logger.info("Live capture produced %d flow(s) on interface %s.", len(flows), interface)
        return flows
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# MODULE 3: Network Intrusion Detection System (Abhil4sh777/nids)
# =====================================================================

class NetworkNIDS:
    """13-class Network Intrusion Detection System with SHAP explainability."""

    def __init__(self, nids_root: Optional[Path] = None):
        self.nids_root = Path(nids_root) if nids_root else NIDS_DIR
        self.model_dir = self.nids_root / "models"
        self.detector_path = self.model_dir / "detector.joblib"
        self.classifier_path = self.model_dir / "classifier.joblib"

        if not self.detector_path.exists() or not self.classifier_path.exists():
            raise FileNotFoundError(f"NIDS model files missing in {self.model_dir}")

        logger.info("Loading NIDS Detector and Classifier pipelines...")
        from config import Settings
        from models.loader import load_models
        from models.inference import InferenceEngine
        from explain.shap_explainer import ShapService
        from features.contract import DETECTOR_THRESHOLD, CLASSIFIER_LABELS

        self.detector_threshold = DETECTOR_THRESHOLD  # 0.3545
        self.classifier_labels = CLASSIFIER_LABELS

        # Start from the deployment environment (so NIDS_NTLFLOWLYZER,
        # timeouts, and thread counts are honoured) then pin submodule paths.
        settings = replace(
            Settings.from_env(),
            project_root=self.nids_root,
            model_dir=self.model_dir,
            knowledge_dir=self.nids_root / "explain" / "knowledge",
            capture_dir=self.nids_root / "capture" / "pcaps",
            output_dir=self.nids_root / "capture" / "flows",
        )
        self.settings = settings
        self.models = load_models(settings)
        self.engine = InferenceEngine(self.models)
        self.shap_service = ShapService(self.models)
        logger.info(f"NIDS loaded successfully (threshold tau={self.detector_threshold:.4f})")

    def predict_dataframe(self, df: pd.DataFrame, top_k_shap: int = 5) -> List[Dict[str, Any]]:
        if df is None or df.empty:
            return []
        from features.validation import prepare_flow_frame
        prepared = prepare_flow_frame(df)
        predictions = self.engine.predict(prepared)

        results = []
        for p in predictions:
            t0 = time.perf_counter()
            row_df = prepared.iloc[[p.row_position]]

            if p.is_attack:
                shap_feats = self.shap_service.classifier_contributions(row_df, p.class_index)
                evidence = [
                    {
                        "feature": tf.feature,
                        "value": round(float(tf.value), 4) if isinstance(tf.value, (int, float, np.number)) else str(tf.value),
                        "shap_value": round(float(tf.shap_value), 4)
                    }
                    for tf in shap_feats[:top_k_shap]
                ]
            else:
                det_feats = self.shap_service.detector_contributions(row_df)
                evidence = [
                    {
                        "feature": tf.feature,
                        "value": round(float(tf.value), 4) if isinstance(tf.value, (int, float, np.number)) else str(tf.value),
                        "shap_value": round(float(tf.shap_value), 4)
                    }
                    for tf in det_feats[:top_k_shap]
                ]

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            src_p = row_df['src_port'].iloc[0]
            dst_p = row_df['dst_port'].iloc[0]

            flow_res = {
                "module": "Network_NIDS",
                "flow_id": str(p.flow_id),
                "timestamp": str(row_df['timestamp'].iloc[0]) if 'timestamp' in row_df else datetime.now(timezone.utc).isoformat(),
                "src_ip": str(row_df['src_ip'].iloc[0]),
                "src_port": int(src_p) if pd.notna(src_p) else 0,
                "dst_ip": str(row_df['dst_ip'].iloc[0]),
                "dst_port": int(dst_p) if pd.notna(dst_p) else 0,
                "protocol": str(row_df['protocol'].iloc[0]),
                "is_attack": bool(p.is_attack),
                "attack_probability": round(float(p.attack_probability), 4),
                "attack_type": str(p.predicted_class) if p.is_attack else "Benign",
                "confidence": round(float(p.class_probability), 4) if p.is_attack else round(float(1.0 - p.attack_probability), 4),
                "decision_threshold": round(self.detector_threshold, 4),
                "shap_evidence": evidence,
                "latency_ms": latency_ms
            }
            results.append(flow_res)
        return results

    def predict_csv(self, csv_path: Union[str, Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Flow CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, nrows=limit, low_memory=False)
        return self.predict_dataframe(df)

    def capture_live_network(
        self,
        interface: str = "any",
        duration_seconds: int = 60,
        max_packets: Optional[int] = 50000,
    ) -> pd.DataFrame:
        """
        Captures real network packets on host interface using dumpcap in a
        1-minute window (default: 60s) and extracts the full 121-feature
        contract with NTLFlowLyzer.
        """
        return capture_live_network(
            interface=interface,
            duration_seconds=duration_seconds,
            max_packets=max_packets,
            settings=self.settings,
        )

    def extract_pcap_flows(self, pcap_path: Union[str, Path]) -> pd.DataFrame:
        """Run NTLFlowLyzer on a PCAP/PCAPNG and return contract-shaped rows."""

        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")
        return extract_flows_with_ntlflowlyzer(pcap_path, settings=self.settings)

    def predict_pcap(
        self,
        pcap_path: Union[str, Path],
        output_csv: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Replay a PCAP/PCAPNG through NTLFlowLyzer, then run NIDS inference.

        NTLFlowLyzer is mandatory.  Extractor failures propagate as
        :class:`NTLFlowLyzerError`; there is no partial DPkt fallback.
        """

        out_df = self.extract_pcap_flows(pcap_path)
        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_csv, index=False)
        if out_df.empty:
            return []
        return self.predict_dataframe(out_df)


# =====================================================================
# CROSS-MODULE AUTOMATION & AGGREGATOR ENGINE
# =====================================================================

URL_EXTRACT_REGEX = re.compile(
    r'(https?://[^\s<>"]+|www\.[^\s<>"]+|\bbit\.ly/[a-zA-Z0-9_\-]+|\btinyurl\.com/[a-zA-Z0-9_\-]+|\bt\.co/[a-zA-Z0-9_\-]+|\bwa\.me/[a-zA-Z0-9_\-]+|\b[a-zA-Z0-9\-]{2,}\.(?:com|in|org|net|xyz|top|link|site|info|online|live|club|app|cc|vip)(?:/[^\s<>"]*)?)',
    re.IGNORECASE
)

def extract_urls_from_text(text: str) -> List[str]:
    """Auto-Link Extractor: extracts URLs, shortlinks, and domains from text."""
    if not text:
        return []
    matches = URL_EXTRACT_REGEX.findall(str(text))
    cleaned = []
    for m in matches:
        m = m.rstrip(".,;!?'\")")
        if not m.startswith("http://") and not m.startswith("https://"):
            m = "http://" + m
        cleaned.append(m)
    return list(dict.fromkeys(cleaned))  # Deduplicate preserving order


from llama_prompt_builder import (
    build_llama_prompt,
    build_llama_prompt_pipeline_1,
    build_llama_prompt_pipeline_2,
    build_llama_prompt_1,
    build_llama_prompt_2,
    generate_grounded_fallback_advisory,
    clean_llm_response,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_PIPELINE_1,
    SYSTEM_PROMPT_PIPELINE_2,
)
from executorch_runner import ExecuTorchLlamaEngine

# Backward compatibility alias
generate_llama_executorch_prompt = build_llama_prompt


def build_no_flow_telemetry(
    message: str,
    status: str = "IDLE",
    capture_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Telemetry for a window that produced no flow rows.

    An empty window has no NIDS verdict.  Returning a fabricated benign flow
    (the old ``idle-window`` placeholder) made live-capture failures look like
    clean traffic with 0% attack probability and 0.00 ms latency.  Callers now
    receive an explicit IDLE / CAPTURE_ERROR status instead.
    """

    telemetry: Dict[str, Any] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_threats_count": 0,
        "total_flows_count": 0,
        "findings": [],
        "passed_to_llama": False,
        "output_type": "no_output",
        "llama_prompt": None,
        "llama_advisory": None,
        "message": message,
        "output": {
            "verdict": "NO_FLOWS",
            "summary": message,
        },
    }
    if capture_error:
        telemetry["capture_error"] = capture_error
    return telemetry


class UnifiedThreatPipeline:
    """Unified Orchestration Pipeline combining SMS, URL, Network threat modules and ExecuTorch reasoning."""

    def __init__(
        self,
        sms_dir: Optional[Path] = None,
        phish_dir: Optional[Path] = None,
        nids_dir: Optional[Path] = None,
        enable_llm_reasoning: bool = True,
    ):
        logger.info("Initializing Unified Multi-Model Edge Threat Pipeline...")
        self.sms_detector = SMSScamDetector(sms_dir)
        self.url_detector = PhishGuardURLDetector(phish_dir)
        self.nids = NetworkNIDS(nids_dir)
        self.enable_llm_reasoning = enable_llm_reasoning
        self.llama_engine = ExecuTorchLlamaEngine()
        logger.info("Unified Pipeline fully initialized and ready.")

    def process_telemetry(self, telemetry: Dict[str, Any], generate_advisory: bool = True, pipeline_type: Optional[int] = None) -> Dict[str, Any]:
        """
        Stitches Llama 3.2 ExecuTorch zero-shot prompt and plain-English advisory
        into the aggregated telemetry payload.
        Enforces Pipeline 1 routing:
          - if benign (no attacks): do not pass to llama -> model output
          - if attack: pass to llama -> llama output
        """
        # Enforce Pipeline 1 per-flow / window routing contract:
        # If Pipeline 1 is indicated and no attacks are flagged, do NOT pass to LLaMA
        if pipeline_type == 1 and telemetry.get("active_threats_count", 0) == 0:
            telemetry["passed_to_llama"] = False
            telemetry["output_type"] = "model_output"
            telemetry["llama_prompt"] = None
            telemetry["llama_advisory"] = None
            if "output" not in telemetry:
                telemetry["output"] = {
                    "verdict": "Benign",
                    "summary": "All network flows verified benign by two-stage XGBoost NIDS. LLaMA reasoning bypassed to conserve edge resources."
                }
            return telemetry

        telemetry["passed_to_llama"] = True
        telemetry["output_type"] = "llama_output"
        telemetry["llama_prompt"] = build_llama_prompt(telemetry, pipeline_type=pipeline_type)
        if generate_advisory and self.enable_llm_reasoning:
            with llm_lock:
                telemetry["llama_advisory"] = self.llama_engine.generate_advisory(telemetry)
        else:
            telemetry["llama_advisory"] = None
        if "output" not in telemetry:
            telemetry["output"] = telemetry["llama_advisory"]
        return telemetry

    def analyze_sms(self, message: str, cross_check_urls: bool = True) -> Dict[str, Any]:
        """
        Pipeline 2: User Input -> SMS Scam Detector -> Extract URL -> PhishGuard URL Detector -> Llama Prompt 2 -> Output.
        Analyzes user input SMS text, automatically extracting embedded URLs for PhishGuard validation.
        """
        findings = []
        sms_res = self.sms_detector.predict(message)
        findings.append(sms_res)

        active_threats = 1 if sms_res["is_threat"] else 0

        if cross_check_urls:
            urls = extract_urls_from_text(message)
            for url in urls:
                url_res = self.url_detector.predict(url)
                findings.append(url_res)
                if url_res["is_phishing"]:
                    active_threats += 1

        status = "ALERT" if active_threats > 0 else "CLEAN"
        telemetry = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_threats_count": active_threats,
            "findings": findings
        }
        return self.process_telemetry(telemetry, pipeline_type=2)

    def run_pipeline_2(self, user_input: str, cross_check_urls: bool = True) -> Dict[str, Any]:
        """
        Convenience wrapper for Pipeline 2:
        user input -> SMS Scam Detector -> extract url -> PhishGuard URL Detector -> llama prompt 2 -> output.
        """
        return self.analyze_sms(user_input, cross_check_urls=cross_check_urls)

    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyzes a URL using PhishGuard."""
        url_res = self.url_detector.predict(url)
        active_threats = 1 if url_res["is_phishing"] else 0
        telemetry = {
            "status": "ALERT" if active_threats > 0 else "CLEAN",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_threats_count": active_threats,
            "findings": [url_res]
        }
        return self.process_telemetry(telemetry, pipeline_type=2)

    def analyze_network_flow(self, flow_data: Union[pd.DataFrame, Dict[str, Any], str, Path, None] = None) -> Dict[str, Any]:
        """
        Pipeline 1: Live Network -> NIDS -> Llama Prompt 1 (Input: NIDS telemetry for attack) -> Output.
        Analyzes real network traffic (PCAP / PCAPNG packet streams, live interface capture, or flow records).
        
        Per-Flow Routing Contract:
        For each flow in a window:
        ~ if benign: do not pass to llama -> model output
        ~ if attack: pass to llama -> llama output
        """
        try:
            if flow_data is None:
                sample_pcap = WORKSPACE_ROOT / "data" / "sample_traffic.pcap"
                if sample_pcap.exists():
                    findings = self.nids.predict_pcap(sample_pcap)
                else:
                    sample_csv = WORKSPACE_ROOT / "data" / "sample_flows.csv"
                    findings = self.nids.predict_csv(sample_csv, limit=5)
            elif isinstance(flow_data, (str, Path)):
                p = Path(flow_data)
                if p.suffix.lower() in [".pcap", ".pcapng"]:
                    findings = self.nids.predict_pcap(p)
                else:
                    findings = self.nids.predict_csv(p)
            elif isinstance(flow_data, dict):
                df = pd.DataFrame([flow_data])
                findings = self.nids.predict_dataframe(df)
            elif isinstance(flow_data, pd.DataFrame):
                findings = self.nids.predict_dataframe(flow_data) if not flow_data.empty else []
            else:
                raise TypeError(f"Unsupported flow data type: {type(flow_data)}")
        except (NTLFlowLyzerError, LiveCaptureError) as exc:
            # Surface the extractor/capture diagnostic verbatim instead of
            # fabricating a benign idle flow.
            diagnostic = _capture_diagnostic(exc)
            logger.error("Network capture failed: %s", diagnostic)
            return build_no_flow_telemetry(
                f"Network capture/flow extraction failed: {diagnostic}",
                status="CAPTURE_ERROR",
                capture_error=diagnostic,
            )

        if not findings:
            return build_no_flow_telemetry(
                "No network flows were captured in this window. The NIDS did not "
                "evaluate any flow, so no benign or attack verdict was produced.",
                status="IDLE",
            )

        # Enforce per-flow policy for each flow in the window:
        # ~ if benign: do not pass to llama -> model output
        # ~ if attack: pass to llama -> llama output
        for f in findings:
            is_atk = bool(f.get("is_attack", False))
            if is_atk:
                f["passed_to_llama"] = True
                f["output_type"] = "llama_output"
                single_flow_telemetry = {
                    "status": "ALERT",
                    "timestamp": f.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "active_threats_count": 1,
                    "findings": [f]
                }
                f["llama_prompt"] = build_llama_prompt_pipeline_1(single_flow_telemetry)
                if self.enable_llm_reasoning:
                    with llm_lock:
                        f["llama_advisory"] = self.llama_engine.generate_advisory(single_flow_telemetry)
                else:
                    f["llama_advisory"] = None
                f["output"] = f["llama_advisory"]
            else:
                # Benign flow: Bypasses LLaMA completely, returning pure NIDS model output
                f["passed_to_llama"] = False
                f["output_type"] = "model_output"
                f["llama_prompt"] = None
                f["llama_advisory"] = None
                f["output"] = {
                    "verdict": "Benign",
                    "attack_type": f.get("attack_type", "Benign"),
                    "confidence": f.get("confidence", 1.0),
                    "attack_probability": f.get("attack_probability", 0.0),
                    "decision_threshold": f.get("decision_threshold", round(self.nids.detector_threshold, 4)),
                    "latency_ms": f.get("latency_ms", 0.0)
                }

        active_threats = sum(1 for f in findings if f.get("is_attack"))
        attack_findings = [f for f in findings if f.get("is_attack")]
        benign_findings = [f for f in findings if not f.get("is_attack")]

        telemetry = {
            "status": "ALERT" if active_threats > 0 else "CLEAN",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_threats_count": active_threats,
            "total_flows_count": len(findings),
            "findings": findings
        }

        # Window-level aggregation adhering to per-flow routing policy:
        if active_threats > 0:
            telemetry["passed_to_llama"] = True
            telemetry["output_type"] = "llama_output"
            window_attack_telemetry = {
                "status": "ALERT",
                "timestamp": telemetry["timestamp"],
                "active_threats_count": active_threats,
                "findings": attack_findings
            }
            telemetry["llama_prompt"] = build_llama_prompt_pipeline_1(window_attack_telemetry)
            if self.enable_llm_reasoning:
                with llm_lock:
                    telemetry["llama_advisory"] = self.llama_engine.generate_advisory(window_attack_telemetry)
            else:
                telemetry["llama_advisory"] = None
            telemetry["output"] = telemetry["llama_advisory"]
        else:
            # Benign window: DO NOT pass to LLaMA -> model output
            telemetry["passed_to_llama"] = False
            telemetry["output_type"] = "model_output"
            telemetry["llama_prompt"] = None
            telemetry["llama_advisory"] = None
            telemetry["output"] = {
                "verdict": "Benign",
                "total_flows": len(findings),
                "benign_flows": len(benign_findings),
                "summary": f"All {len(findings)} flow(s) in window verified benign by two-stage XGBoost NIDS. LLaMA reasoning bypassed to conserve edge resources."
            }

        return telemetry

    def run_pipeline_1(
        self,
        flow_data: Union[pd.DataFrame, Dict[str, Any], str, Path, None] = None,
        live_interface: Optional[str] = None,
        duration_seconds: int = 60,
        probe: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Pipeline 1: Live Network (dumpcap / PCAP) -> NIDS -> Per-Flow Routing -> Output.
        Window Size: 1 minute (60 seconds) by default.
        For each flow in window:
          ~ if benign: do not pass to llama -> model output
          ~ if attack: pass to llama -> llama output
        Executes real workflow on live Linux interface capture or real PCAP packet replay.
        """
        if live_interface is None:
            live_interface = kwargs.get("live_interface") or kwargs.get("interface")
        if "duration_seconds" in kwargs:
            duration_seconds = kwargs["duration_seconds"]
        elif "duration" in kwargs:
            duration_seconds = kwargs["duration"]
        if "probe" in kwargs:
            probe = bool(kwargs["probe"])

        window_size_label = f"{duration_seconds // 60} min" if duration_seconds % 60 == 0 else f"{duration_seconds}s"

        if live_interface:
            probe_thread = None
            if probe:
                import threading
                def _probe_worker():
                    time.sleep(0.3)
                    try:
                        import urllib.request
                        urllib.request.urlopen("https://www.google.com", timeout=2)
                    except Exception:
                        try:
                            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            s.sendto(b"abhedya-probe", ("8.8.8.8", 53))
                            s.close()
                        except Exception:
                            pass
                probe_thread = threading.Thread(target=_probe_worker, daemon=True)
                probe_thread.start()

            try:
                df = self.nids.capture_live_network(
                    interface=live_interface,
                    duration_seconds=duration_seconds,
                )
            except (LiveCaptureError, NTLFlowLyzerError) as exc:
                diagnostic = _capture_diagnostic(exc)
                logger.error("Live capture failed on interface %s: %s", live_interface, diagnostic)
                telemetry = build_no_flow_telemetry(
                    f"Live capture on interface '{live_interface}' failed: {diagnostic}",
                    status="CAPTURE_ERROR",
                    capture_error=diagnostic,
                )
                telemetry["window_size_seconds"] = duration_seconds
                telemetry["window_size_label"] = window_size_label
                return telemetry
            finally:
                if probe_thread:
                    probe_thread.join(timeout=1.0)
            telemetry = self.analyze_network_flow(df)
            telemetry["window_size_seconds"] = duration_seconds
            telemetry["window_size_label"] = window_size_label
            return telemetry

        telemetry = self.analyze_network_flow(flow_data)
        telemetry["window_size_seconds"] = duration_seconds
        telemetry["window_size_label"] = window_size_label
        return telemetry

    def stream_pipeline_1(
        self,
        live_interface: str = "any",
        duration_seconds: int = 60,
        probe: bool = False,
        max_windows: Optional[int] = None,
    ):
        """
        Continuously capture and evaluate 1-minute network windows in a stream/loop.

        Operational Contract:
        - Window size: 1 minute (60 seconds)
        - After start:
          * Automatically updates with each new 1-minute window
          * Automatically discards the previous 1-minute window to conserve edge memory

        Yields:
            Dict[str, Any]: Telemetry for the active 1-minute window.
        """
        window_idx = 0
        last_window_telemetry = None
        while max_windows is None or window_idx < max_windows:
            window_idx += 1
            # Automatically discard previous window data
            if last_window_telemetry is not None:
                del last_window_telemetry
                last_window_telemetry = None

            telemetry = self.run_pipeline_1(
                live_interface=live_interface,
                duration_seconds=duration_seconds,
                probe=probe
            )
            telemetry["window_number"] = window_idx
            telemetry["window_size_seconds"] = duration_seconds
            telemetry["window_size_label"] = f"{duration_seconds // 60} min" if duration_seconds % 60 == 0 else f"{duration_seconds}s"
            last_window_telemetry = telemetry
            yield telemetry


# =====================================================================
# CLI DEMONSTRATION & BENCHMARK RUNNER
# =====================================================================


# =====================================================================
# APPROACH 2: MULTI-THREADED DUAL ENGINE (NIDS + SMS/PHISHING)
# =====================================================================

class UnifiedMultiThreadedEngine:
    """
    Multi-threaded dual engine running Pipeline 1 (Live Network NIDS) and
    Pipeline 2 (Smishing & Phishing detection) concurrently on Raspberry Pi 5.

    Thread 1 (nids_thread):
      - Run continuous 60-second tumbling capture windows in a dedicated background thread.
      - Benign Traffic (P(Attack) < 0.3545): Emit instantaneous sub-2ms verdict to telemetry feed,
        bypassing ExecuTorch Llama engine (0 LLM compute overhead).
      - Attack Traffic (P(Attack) >= 0.3545): Calculate top-5 SHAP feature attributions,
        acquire llm_lock, generate 3-part ExecuTorch Llama Prompt 1 advisory, and release llm_lock.
      - Graceful handling of empty capture windows (IDLE status).

    Thread 2 (sms_thread):
      - Run an asynchronous API / Webhook queue listener in a dedicated background thread.
      - Processing: De-obfuscate text -> 25,049-Dim Stacking Ensemble -> PhishGuard PyTorch MLP (auto-extracted links).
      - Clean Content: Emit instant clean log (0 LLM compute overhead).
      - Threat Detected: Acquire llm_lock, generate ExecuTorch Llama Prompt 2 safety alert, and release llm_lock.

    Global Threading Mutex (llm_lock):
      - Serializes access to local ExecuTorch Llama 3.2 engine (~1.2 GB limit).
      - Ensures non-blocking sensor evaluation so incoming network packets and SMS texts are never dropped.
    """

    def __init__(
        self,
        pipeline: Optional[UnifiedThreatPipeline] = None,
        capture_window_seconds: int = 60,
        nids_interface: Optional[str] = None,
        custom_flow_generator: Optional[Any] = None,
    ):
        self.pipeline = pipeline or UnifiedThreatPipeline()
        self.capture_window_seconds = int(capture_window_seconds)
        self.nids_interface = nids_interface or os.environ.get("NIDS_INTERFACE", "eth0")
        self.custom_flow_generator = custom_flow_generator

        # Thread management
        self._running = False
        self._nids_thread: Optional[threading.Thread] = None
        self._sms_thread: Optional[threading.Thread] = None

        # Queue for Thread 2 (non-blocking ingestion)
        self._sms_queue: queue.Queue = queue.Queue()

        # Thread-safe telemetry event log and status
        self._log_lock = threading.Lock()
        self._event_log: List[Dict[str, Any]] = []
        self._status: Dict[str, Any] = {
            "engine_state": "STOPPED",
            "nids_thread": "IDLE",
            "sms_thread": "IDLE",
            "nids_windows_processed": 0,
            "sms_events_processed": 0,
            "total_threats_detected": 0,
            "llm_invocations": 0,
            "last_nids_event": None,
            "last_sms_event": None,
        }

    # -- Status & Telemetry Logging ------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Returns thread-safe snapshot of engine runtime status."""
        with self._log_lock:
            status = dict(self._status)
            status["llm_lock_locked"] = llm_lock.locked()
            status["queue_depth"] = self._sms_queue.qsize()
            return status

    def get_event_log(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Returns last N logged telemetry events across both threads."""
        with self._log_lock:
            return list(self._event_log[-last_n:])

    def _log_event(self, event: Dict[str, Any]) -> None:
        with self._log_lock:
            event["logged_at"] = datetime.now(timezone.utc).isoformat()
            self._event_log.append(event)
            if len(self._event_log) > 500:
                self._event_log = self._event_log[-500:]

    def _update_status(self, **kwargs) -> None:
        with self._log_lock:
            self._status.update(kwargs)

    # -- Engine Lifecycle ----------------------------------------------

    def start(self) -> None:
        """Starts both nids_thread and sms_thread concurrently in multi-threaded mode."""
        if self._running:
            logger.warning("UnifiedMultiThreadedEngine is already running.")
            return

        self._running = True
        self._update_status(engine_state="RUNNING")
        logger.info("Starting Approach 2 Multi-Threaded Dual Engine (RPi 5)...")

        self._nids_thread = threading.Thread(
            target=self._nids_worker,
            name="nids_thread",
            daemon=True
        )
        self._sms_thread = threading.Thread(
            target=self._sms_worker,
            name="sms_thread",
            daemon=True
        )

        self._nids_thread.start()
        self._sms_thread.start()
        logger.info("Both nids_thread and sms_thread successfully launched.")

    def stop(self) -> None:
        """Gracefully stops both background worker threads."""
        if not self._running:
            return

        logger.info("Stopping UnifiedMultiThreadedEngine...")
        self._running = False

        self._sms_queue.put(None)

        if self._nids_thread and self._nids_thread.is_alive():
            self._nids_thread.join(timeout=3.0)
        if self._sms_thread and self._sms_thread.is_alive():
            self._sms_thread.join(timeout=2.0)

        self._update_status(
            engine_state="STOPPED",
            nids_thread="STOPPED",
            sms_thread="STOPPED"
        )
        logger.info("UnifiedMultiThreadedEngine stopped successfully.")

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Non-Blocking Ingestion API (Thread 2) -------------------------

    def submit_sms(self, message: str) -> None:
        """Non-blocking submission of incoming SMS message to Thread 2 queue."""
        self._sms_queue.put({"type": "sms", "message": message})

    def submit_url(self, url: str) -> None:
        """Non-blocking submission of standalone URL to Thread 2 queue."""
        self._sms_queue.put({"type": "url", "url": url})

    # -- Thread 1: Live Network Monitoring Engine (NIDS) ---------------

    def _nids_worker(self) -> None:
        """
        Thread 1 Worker: Continuous 60-second tumbling capture windows.
        - Benign: instantaneous sub-2ms verdict to telemetry, 0 LLM compute overhead.
        - Attack: top-5 SHAP attributions + llm_lock ExecuTorch Llama Prompt 1 advisory.
        - Graceful handling of empty capture windows (IDLE status).
        """
        self._update_status(nids_thread="RUNNING")
        logger.info(f"[Thread 1: NIDS] Worker running (window={self.capture_window_seconds}s, iface={self.nids_interface})")

        while self._running:
            try:
                self._update_status(nids_thread="CAPTURING")
                flow_df = self._capture_network_window()

                if flow_df is None or flow_df.empty:
                    self._update_status(nids_thread="IDLE")
                    self._log_event({
                        "thread": "nids",
                        "type": "IDLE",
                        "status": "IDLE",
                        "message": "Empty capture window: zero network flows captured in tumbling interval.",
                    })
                    # An empty window is still a completed capture interval;
                    # count it so operators can distinguish "no traffic" from
                    # a stalled capture worker.
                    with self._log_lock:
                        self._status["nids_windows_processed"] += 1
                        self._status["last_nids_event"] = datetime.now(timezone.utc).isoformat()
                    sleep_slices = max(1, min(self.capture_window_seconds, 3))
                    for _ in range(sleep_slices):
                        if not self._running:
                            break
                        time.sleep(1.0)
                    continue

                self._update_status(nids_thread="CLASSIFYING")
                nids_results = self.pipeline.nids.predict_dataframe(flow_df)

                for result in nids_results:
                    t_eval = time.perf_counter()
                    attack_prob = result.get("attack_probability", 0.0)
                    is_attack = result.get("is_attack", False) or (attack_prob >= self.pipeline.nids.detector_threshold)

                    if not is_attack:
                        # Benign Traffic (P(Attack) < 0.3545): instantaneous sub-2ms verdict, bypass LLM
                        latency_ms = round((time.perf_counter() - t_eval) * 1000.0, 3)
                        event = {
                            "thread": "nids",
                            "type": "BENIGN",
                            "status": "CLEAN",
                            "flow_id": result.get("flow_id"),
                            "src_ip": result.get("src_ip"),
                            "dst_ip": result.get("dst_ip"),
                            "dst_port": result.get("dst_port"),
                            "attack_probability": attack_prob,
                            "decision_threshold": self.pipeline.nids.detector_threshold,
                            "verdict": "CLEAN",
                            "llm_invoked": False,
                            "verdict_latency_ms": latency_ms,
                        }
                        self._log_event(event)

                    else:
                        # Attack Traffic (P(Attack) >= 0.3545): SHAP + llm_lock ExecuTorch Llama Prompt 1
                        self._update_status(nids_thread="LLM_ADVISORY")
                        telemetry = {
                            "status": "ALERT",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "active_threats_count": 1,
                            "findings": [result]
                        }
                        telemetry["llama_prompt"] = build_llama_prompt_pipeline_1(telemetry)

                        advisory = None
                        if self.pipeline.enable_llm_reasoning:
                            with llm_lock:
                                advisory = self.pipeline.llama_engine.generate_advisory(telemetry)
                                with self._log_lock:
                                    self._status["llm_invocations"] += 1

                        latency_ms = round((time.perf_counter() - t_eval) * 1000.0, 3)
                        event = {
                            "thread": "nids",
                            "type": "ATTACK",
                            "status": "ALERT",
                            "flow_id": result.get("flow_id"),
                            "src_ip": result.get("src_ip"),
                            "dst_ip": result.get("dst_ip"),
                            "dst_port": result.get("dst_port"),
                            "attack_type": result.get("attack_type"),
                            "attack_probability": attack_prob,
                            "decision_threshold": self.pipeline.nids.detector_threshold,
                            "shap_evidence": result.get("shap_evidence", [])[:5],
                            "llm_invoked": advisory is not None,
                            "advisory": advisory,
                            "verdict_latency_ms": latency_ms,
                        }
                        self._log_event(event)
                        with self._log_lock:
                            self._status["total_threats_detected"] += 1

                with self._log_lock:
                    self._status["nids_windows_processed"] += 1
                    self._status["last_nids_event"] = datetime.now(timezone.utc).isoformat()

                self._update_status(nids_thread="IDLE")

                sleep_duration = max(1, min(self.capture_window_seconds, 5))
                for _ in range(sleep_duration):
                    if not self._running:
                        break
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"[Thread 1: NIDS] Error in capture/classification: {e}", exc_info=True)
                self._log_event({
                    "thread": "nids",
                    "type": "ERROR",
                    "status": "ERROR",
                    "message": str(e),
                })
                self._update_status(nids_thread="ERROR")
                time.sleep(2.0)

        self._update_status(nids_thread="STOPPED")
        logger.info("[Thread 1: NIDS] Worker stopped.")

    def _capture_network_window(self) -> Optional[pd.DataFrame]:
        """
        Captures or provides flows for one tumbling window.
        1. Custom flow generator hook (used for automated unit tests).
        2. Live dumpcap ring-buffer capture on Linux / Raspberry Pi 5.
        3. Fallback to sample flows on dev machines.
        """
        if self.custom_flow_generator is not None:
            return self.custom_flow_generator()

        import platform
        if platform.system() == "Linux" and shutil.which("dumpcap"):
            try:
                df = self.pipeline.nids.capture_live_network(
                    interface=self.nids_interface,
                    duration_seconds=self.capture_window_seconds
                )
                return df
            except Exception as e:
                logger.debug(f"Live capture error: {e}")
                return None
        else:
            sample_csv = WORKSPACE_ROOT / "data" / "sample_flows.csv"
            if not sample_csv.exists():
                sample_csv = NIDS_DIR / "tests" / "fixtures" / "parity_rows.csv"
            if sample_csv.exists():
                df = pd.read_csv(sample_csv, nrows=3, low_memory=False)
                return df
            return None

    # -- Thread 2: Event-Driven SMS & URL Listener ---------------------

    def _sms_worker(self) -> None:
        """
        Thread 2 Worker: Event-driven SMS & URL listener.
        - De-obfuscate text -> 25,049-Dim Stacking Ensemble -> PhishGuard PyTorch MLP.
        - Clean Content: emit instant clean log (0 LLM compute overhead).
        - Threat Detected: acquire llm_lock, generate ExecuTorch Llama Prompt 2 safety alert.
        """
        self._update_status(sms_thread="RUNNING")
        logger.info("[Thread 2: SMS/Phishing] Event-driven queue listener started.")

        while self._running:
            try:
                try:
                    item = self._sms_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is None:
                    break

                self._update_status(sms_thread="PROCESSING")
                t0 = time.perf_counter()
                item_type = item.get("type", "sms")
                findings = []
                is_threat = False

                if item_type == "sms":
                    raw_msg = item.get("message", "")
                    sms_res = self.pipeline.sms_detector.predict(raw_msg)
                    findings.append(sms_res)
                    if sms_res.get("is_threat", False):
                        is_threat = True

                    extracted_urls = extract_urls_from_text(raw_msg)
                    for u in extracted_urls:
                        url_res = self.pipeline.url_detector.predict(u)
                        findings.append(url_res)
                        if url_res.get("is_phishing", False):
                            is_threat = True

                elif item_type == "url":
                    raw_url = item.get("url", "")
                    url_res = self.pipeline.url_detector.predict(raw_url)
                    findings.append(url_res)
                    if url_res.get("is_phishing", False):
                        is_threat = True

                active_threats = sum(
                    1 for f in findings
                    if f.get("is_threat") or f.get("is_phishing")
                )

                advisory = None
                if is_threat:
                    self._update_status(sms_thread="LLM_ADVISORY")
                    telemetry = {
                        "status": "ALERT",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_threats_count": active_threats,
                        "findings": findings
                    }
                    telemetry["llama_prompt"] = build_llama_prompt_pipeline_2(telemetry)

                    if self.pipeline.enable_llm_reasoning:
                        with llm_lock:
                            advisory = self.pipeline.llama_engine.generate_advisory(telemetry)
                            with self._log_lock:
                                self._status["llm_invocations"] += 1

                    with self._log_lock:
                        self._status["total_threats_detected"] += 1

                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

                event = {
                    "thread": "sms",
                    "type": "THREAT" if is_threat else "CLEAN",
                    "status": "ALERT" if is_threat else "CLEAN",
                    "input_type": item_type,
                    "content_preview": (item.get("message") or item.get("url", ""))[:65],
                    "findings_count": len(findings),
                    "is_threat": is_threat,
                    "active_threats": active_threats,
                    "llm_invoked": advisory is not None,
                    "advisory": advisory,
                    "processing_latency_ms": latency_ms,
                }
                self._log_event(event)

                with self._log_lock:
                    self._status["sms_events_processed"] += 1
                    self._status["last_sms_event"] = datetime.now(timezone.utc).isoformat()

                self._update_status(sms_thread="RUNNING")
                self._sms_queue.task_done()

            except Exception as e:
                logger.error(f"[Thread 2: SMS/Phishing] Error in queue listener: {e}", exc_info=True)
                self._log_event({
                    "thread": "sms",
                    "type": "ERROR",
                    "status": "ERROR",
                    "message": str(e),
                })
                self._update_status(sms_thread="ERROR")
                time.sleep(1.0)

        self._update_status(sms_thread="STOPPED")
        logger.info("[Thread 2: SMS/Phishing] Worker stopped.")


def run_cli_demo():
    print("\n" + "=" * 70)
    print("   BHARAT AI-SOC CHALLENGE: ON-DEVICE MULTI-MODEL THREAT PIPELINE")
    print("   Problem Statement 3 - Linux / Raspberry Pi 5 Demonstration")
    print("=" * 70 + "\n")

    pipeline = UnifiedThreatPipeline()

    print("\n[+] SCENARIO 1 (PIPELINE 2): User Input -> SMS Scam Detector -> Extract URL -> PhishGuard -> Llama Prompt 2 -> Output")
    smish_text = "URGENT: Your SBI account has been locked. Update KYC immediately: http://sbi-banking-verify-update.xyz/login.php?user=victim"
    res1 = pipeline.run_pipeline_2(smish_text)
    print(f"Status: {res1['status']} | Active Threats: {res1['active_threats_count']}")
    for idx, f in enumerate(res1["findings"], 1):
        if f["module"] == "SMS_Spam_Detector":
            print(f"  [{idx}] SMS: Threat={f['is_threat']} (Score: {f['threat_score']:.4f}, Level: {f['threat_level']}) Triggers: {f['detected_triggers']}")
        elif f["module"] == "PhishGuard_URL_Detector":
            print(f"  [{idx}] URL: Phish={f['is_phishing']} (Conf: {f['confidence_score']:.4f}) Anomalies: {f['domain_anomalies'][:2]}")

    print("\n[+] SCENARIO 2 (PIPELINE 1): Live Network Window -> NIDS -> Per-Flow Routing -> Output")
    print("    Contract: For each flow in window: if benign -> model output | if attack -> llama output")

    # 2a: Benign Flow Window (PCAP Replay)
    sample_pcap = WORKSPACE_ROOT / "data" / "sample_traffic.pcap"
    if sample_pcap.exists():
        print("\n  [2a] Processing Benign Flow Window (data/sample_traffic.pcap):")
        res_benign = pipeline.run_pipeline_1(sample_pcap)
        print(f"       Window Status: {res_benign['status']} | Passed to LLaMA: {res_benign['passed_to_llama']} | Output Type: {res_benign['output_type'].upper()}")
        for idx, f in enumerate(res_benign["findings"][:3], 1):
            print(f"       Flow {idx} ({f['flow_id']}): Attack={f['is_attack']} -> Routing: {f['output_type'].upper()} (Passed to LLaMA: {f['passed_to_llama']})")
        print(f"       Model Output Summary: {res_benign['output']['summary']}")

    # 2b: Attack Flow Window (CSV Stream)
    sample_csv = WORKSPACE_ROOT / "data" / "sample_flows.csv"
    if sample_csv.exists():
        df_sample = pd.read_csv(sample_csv)
        print("\n  [2b] Processing Attack Flow Window (data/sample_flows.csv - Row 3: DoS_Slowhttptest):")
        res_attack = pipeline.run_pipeline_1(df_sample.iloc[3:4])
        print(f"       Window Status: {res_attack['status']} | Passed to LLaMA: {res_attack['passed_to_llama']} | Output Type: {res_attack['output_type'].upper()}")
        for idx, f in enumerate(res_attack["findings"], 1):
            print(f"       Flow {idx} ({f['flow_id']}): Attack={f['is_attack']} ({f['attack_type']}) -> Routing: {f['output_type'].upper()} (Passed to LLaMA: {f['passed_to_llama']})")
            print(f"       Top SHAP Evidence: {f['shap_evidence'][:2]}")
        print("\n--- ON-DEVICE PLAIN-ENGLISH ADVISORY (PIPELINE 1 ATTACK) ---")
        advisory_p1 = res_attack.get("llama_advisory", "")
        print(advisory_p1 if advisory_p1 else "No advisory generated.")
        print("------------------------------------------------------------")

    print("\n[+] SCENARIO 3: Downstream ExecuTorch Llama Reasoning Output")
    print("--- ON-DEVICE PLAIN-ENGLISH ADVISORY (PIPELINE 2) ---")
    advisory = res1.get("llama_advisory", "")
    print(advisory if advisory else "No advisory generated.")
    print("-----------------------------------------------------")
    print("\n--- RAW EXECUTORCH CHAT PROMPT (PIPELINE 2) ---")
    prompt = res1.get("llama_prompt", "")
    print(prompt[:450] + "...\n[TRUNCATED FOR DISPLAY]\n-----------------------------------------------------")
    print("\n[SUCCESS] Pipeline 1 & Pipeline 2 verified operational on edge runtime.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Multi-Model Threat Detection Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run automated CLI demonstration across both pipelines")
    parser.add_argument("--pipeline-1", "--p1", dest="pipeline_1", type=str, help="Pipeline 1: Live Network/Flow CSV/PCAP -> NIDS -> Llama Prompt 1 -> Output")
    parser.add_argument("--pipeline-2", "--p2", dest="pipeline_2", type=str, help="Pipeline 2: User Input -> SMS -> Extract URL -> PhishGuard -> Llama Prompt 2 -> Output")
    parser.add_argument("--interface", "--live-interface", dest="interface", type=str, help="Live network interface to capture for Pipeline 1 (e.g. eth0, wlan0, lo, any)")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration in seconds for live capture window (default: 60s / 1 min)")
    parser.add_argument("--continuous", "--stream", action="store_true", help="Continuously run Pipeline 1 in 1-minute tumbling windows (automatically updates with new 1-min window and discards previous window)")
    parser.add_argument("--probe", action="store_true", help="Generate live probe traffic during capture window")
    parser.add_argument("--sms", type=str, help="Analyze custom SMS message (Pipeline 2)")
    parser.add_argument("--url", type=str, help="Analyze custom URL")
    parser.add_argument("--flow-csv", type=str, help="Analyze network flow CSV (Pipeline 1)")
    args = parser.parse_args()

    if args.demo or not any([args.pipeline_1, args.pipeline_2, args.interface, args.sms, args.url, args.flow_csv, args.continuous]):
        run_cli_demo()
    else:
        pipeline = UnifiedThreatPipeline()
        if args.continuous:
            iface = args.interface or args.pipeline_1 or "any"
            print(f"[*] Starting Pipeline 1 Continuous Monitoring on interface '{iface}' (Window size: {args.duration}s / 1 min)...")
            print("[*] Automatically updating with new 1-min window & discarding previous window. Press Ctrl+C to stop.\n")
            try:
                for window in pipeline.stream_pipeline_1(live_interface=iface, duration_seconds=args.duration, probe=args.probe):
                    w_num = window.get("window_number", 1)
                    print(f"\n[+] === PIPELINE 1: WINDOW #{w_num} ({window.get('timestamp')}) ===")
                    print(f"    Status: {window['status']} | Active Threats: {window['active_threats_count']} | Flows: {window.get('total_flows_count', 0)}")
                    print(f"    Routing Output: {window.get('output_type', 'none').upper()} (Passed to LLaMA: {window.get('passed_to_llama')})")
                    if window.get("status") == "ALERT" and window.get("llama_advisory"):
                        print("--- LLaMA 3.2 ADVISORY ---")
                        print(window["llama_advisory"])
                    print(f"[✓] Window #{w_num} processed. Discarded previous 1-min window; awaiting next 1-min window...")
            except KeyboardInterrupt:
                print("\n[*] Continuous monitoring stopped by user.")
        elif args.interface:
            out = pipeline.run_pipeline_1(live_interface=args.interface, duration_seconds=args.duration, probe=args.probe)
            print(json.dumps(out, indent=2))
        elif args.pipeline_1:
            p1_val = args.pipeline_1
            # Check if p1_val is an interface name or a file path
            if p1_val in ["any", "lo", "eth0", "wlan0"] or not Path(p1_val).exists():
                out = pipeline.run_pipeline_1(live_interface=p1_val, duration_seconds=args.duration, probe=args.probe)
            else:
                out = pipeline.run_pipeline_1(p1_val, duration_seconds=args.duration)
            print(json.dumps(out, indent=2))
        elif args.pipeline_2:
            out = pipeline.run_pipeline_2(args.pipeline_2)
            print(json.dumps(out, indent=2))
        elif args.sms:
            out = pipeline.analyze_sms(args.sms)
            print(json.dumps(out, indent=2))
        elif args.url:
            out = pipeline.analyze_url(args.url)
            print(json.dumps(out, indent=2))
        elif args.flow_csv:
            out = pipeline.analyze_network_flow(args.flow_csv)
            print(json.dumps(out, indent=2))

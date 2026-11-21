"""
Bharat AI-SoC: Sub-System Threat Detection Modules Package
Integrates:
- modules.sms_spam_detector: Sub-millisecond SMS text threat classifier (A-spark-of-fire)
- modules.phishguard: Deep URL structural and domain feature analyzer (SwastikSaha)
- modules.nids: Real-time network flow monitor and 13-class attack classifier with SHAP (Abhil4sh777)
"""

from __future__ import annotations

import sys
from pathlib import Path

MODULES_DIR = Path(__file__).resolve().parent
SMS_DIR = MODULES_DIR / "sms_spam_detector"
PHISH_DIR = MODULES_DIR / "phishguard"
NIDS_DIR = MODULES_DIR / "nids"

# Configure sys.path so internal relative imports inside each submodule resolve seamlessly
for p in [MODULES_DIR, SMS_DIR, PHISH_DIR, NIDS_DIR]:
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

__all__ = ["MODULES_DIR", "SMS_DIR", "PHISH_DIR", "NIDS_DIR"]

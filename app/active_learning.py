# Active learning strategies
# app/active_learning.py
"""
Active-learning helper: log low-confidence examples for later labeling.
Appends JSONL records to data/low_confidence.jsonl
"""
import json
import os
from typing import Dict

LOG_PATH = os.path.join("data", "low_confidence.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log_low_confidence(record: Dict):
    """
    record should contain keys: text, candidate_labels, scores, chosen_label, timestamp
    """
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

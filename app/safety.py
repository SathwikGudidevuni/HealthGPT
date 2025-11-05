# Safety mechanisms
# app/safety.py
"""
Safety utilities: red-flag detection and escalation helper.
"""
import re
from typing import List, Dict


RED_FLAG_PATTERNS = [
    r"\bdifficulty breathing\b",
    r"\bshortness of breath\b",
    r"\bchest pain\b",
    r"\bsevere bleeding\b",
    r"\bpass(ed)? out\b",
    r"\blose consciousness\b",
    r"\bunable to move\b",
    r"\bblue lips\b",
]


def detect_red_flags(text: str) -> List[str]:
    lowered = text.lower()
    hits = []
    for p in RED_FLAG_PATTERNS:
        if re.search(p, lowered):
            hits.append(p)
    return hits


def escalate_response(flags: List[str]) -> str:
    if not flags:
        return ""
    return (
        "Your description contains symptoms that may be serious (e.g., difficulty breathing or chest pain). "
        "Please seek emergency medical care or call your local emergency number immediately. "
        "Do not delay."
    )

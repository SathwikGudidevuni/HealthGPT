# Clarification implementation
# app/clarifier.py
"""
Clarifier module: creates short clarifying questions when the pipeline
is not confident about the detected intent or when multiple high-scoring
labels are close in score.

Public functions:
- build_clarification(candidate_labels, scores, text) -> str
- build_slot_clarification(structured) -> str
"""
from typing import List
import difflib


def build_clarification(candidate_labels: List[str], scores: List[float], text: str) -> str:
    """
    Given candidate labels and their scores (same order), produce a short
    clarifying question. If the top two labels are close, offer them as choices.
    """
    if not candidate_labels or not scores:
        return "Could you clarify what you mean by that?"

    # pair and sort by score
    pairs = sorted(zip(candidate_labels, scores), key=lambda x: x[1], reverse=True)
    top_label, top_score = pairs[0]
    if len(pairs) == 1:
        return f"Can you clarify if your question is about '{top_label}' or something else?"

    second_label, second_score = pairs[1]

    # if scores are close, present a binary clarification
    if top_score - second_score < 0.15:
        # try to produce human-friendly labels (improve readability)
        def nice(l):
            return l.replace("_", " ")
        return (
            f"Do you mean {nice(top_label)} or {nice(second_label)}? "
            "Could you specify which one and provide a little more detail (duration, severity)?"
        )

    # Otherwise ask a slot clarification (duration/severity) if likely a symptom query
    if top_label in ("symptom_query", "medication_query"):
        return (
            "Could you tell me how long you've had these symptoms and whether they are mild, moderate, or severe?"
        )

    # generic fallback
    return "Could you clarify your question a bit more so I can help you better?"


def build_slot_clarification(structured: dict) -> str:
    """
    Look at structured dict and ask targeted slot-filling questions.
    """
    missing = []
    if not structured.get("symptoms"):
        missing.append("what symptoms you have")
    if not structured.get("duration"):
        missing.append("how long you've had them")
    if not structured.get("severity"):
        missing.append("whether symptoms are mild, moderate, or severe")

    if not missing:
        return "Could you provide more details about what you're concerned about?"
    # join nicely
    if len(missing) == 1:
        q = missing[0]
    else:
        q = ", and ".join(missing)
    return f"Could you tell me {q}?"

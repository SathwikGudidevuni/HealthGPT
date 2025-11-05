# Pipeline implementation
# app/pipeline.py
"""
HealthPipeline (with clarification, confidence, dialogue state, retrieval & safety)
- Computes intent confidence (classifier softmax or zero-shot scores)
- If confidence < threshold, returns a clarifying question (no generation) and logs the example
- Detects red flags and escalates instead of generating
- Optionally retrieves grounding snippets and includes them in the generation prompt
- Maintains simple in-memory session state for follow-ups
"""
import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from models.comprehension_model import ComprehensionModel
from models.generation_model import GenerationModel
from app.clarifier import build_clarification, build_slot_clarification
from app.dialogue import DialogueManager
from app.retrieval import Retriever
from app.safety import detect_red_flags, escalate_response
from app.active_learning import log_low_confidence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HealthPipeline:
    def __init__(
        self,
        comp_model_name: str = None,
        gen_model_name: str = None,
        comp_classifier_path: str = None,
        device: str = None,
        confidence_threshold: float = 0.65,
        retriever_path: str = None,
    ):
        logger.info("[HealthPipeline] Initializing models...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize comprehension and generation models
        self.comprehension = ComprehensionModel(
            model_name=comp_model_name,
            classifier_path=comp_classifier_path,
            device=self.device,
        )
        self.generation = GenerationModel(
            model_name=gen_model_name,
            device=self.device,
        )

        # Dialogue and retrieval modules
        self.dialogue = DialogueManager()
        self.confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", confidence_threshold))
        self.retriever = Retriever(rules_path=retriever_path)
        logger.info("[HealthPipeline] Models loaded.")

    def _compute_intent_confidence(self, text: str):
        """
        Returns (chosen_label, confidence, candidate_labels, scores)
        Uses the classifier if available; otherwise uses zero-shot pipeline scores.
        """
        if getattr(self.comprehension, "use_classifier", False) and getattr(self.comprehension, "classifier_model", None) is not None:
            try:
                tokenizer = self.comprehension.tokenizer
                model = self.comprehension.classifier_model
                inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                id2label = getattr(self.comprehension, "id2label", None)
                labels = [id2label[i] for i in range(len(probs))] if id2label else [str(i) for i in range(len(probs))]
                top_idx = int(probs.argmax())
                return labels[top_idx], float(probs[top_idx]), labels, [float(x) for x in probs.tolist()]
            except Exception as e:
                logger.warning(f"[HealthPipeline] classifier confidence compute failed: {e}")

        # zero-shot fallback
        try:
            zs = getattr(self.comprehension, "zs_pipeline", None)
            candidate_labels = getattr(
                self.comprehension, "intent_labels",
                ["symptom_query", "general_question", "medication_query", "appointment_request", "other"]
            )
            if zs:
                zs_out = zs(text, candidate_labels=candidate_labels, multi_label=False)
                labels = zs_out.get("labels", candidate_labels)
                scores = zs_out.get("scores", [])
                top_label = labels[0] if labels else "other"
                top_score = float(scores[0]) if scores else 0.0
                return top_label, top_score, labels, [float(s) for s in scores]
        except Exception as e:
            logger.warning(f"[HealthPipeline] zero-shot confidence compute failed: {e}")

        return "other", 0.0, ["other"], [0.0]

    def run(self, text: str, session_id: Optional[str] = None) -> Dict:
        """
        Full pipeline execution
        """
        ts = time.time()
        structured = self.comprehension.analyze_input(text)
        chosen_label, confidence, candidate_labels, scores = self._compute_intent_confidence(text)
        structured["intent"] = chosen_label
        structured["confidence"] = confidence

        # Dialogue handling
        if session_id:
            session = self.dialogue.get(session_id)
            session.add_user(text)
        else:
            session = None

        # Red-flag detection
        flags = detect_red_flags(text)
        if flags:
            msg = escalate_response(flags)
            if session:
                session.add_assistant(msg)
            return {
                "response": msg,
                "context": structured,
                "confidence": confidence,
                "clarify": False,
                "escalate": True,
                "grounding": []
            }

        # Clarification if low confidence
        if confidence < self.confidence_threshold:
            clarify_q = build_clarification(candidate_labels, scores, text)
            slot_q = build_slot_clarification(structured)
            q = slot_q if structured.get("intent") == "symptom_query" and ("duration" not in structured or not structured.get("duration")) else clarify_q
            try:
                log_low_confidence({
                    "text": text,
                    "candidate_labels": candidate_labels,
                    "scores": scores,
                    "chosen_label": chosen_label,
                    "timestamp": int(ts),
                })
            except Exception:
                logger.exception("Failed to log low-confidence sample")
            if session:
                session.add_assistant(q)
            return {
                "response": q,
                "context": structured,
                "confidence": confidence,
                "clarify": True,
                "escalate": False,
                "grounding": []
            }

        # Retrieval / grounding
        grounding = self.retriever.retrieve(structured)
        session_summary = session.summary() if session else ""
        structured["grounding"] = grounding
        structured["session_summary"] = session_summary

        # Generation
        response = self.generation.generate_response(structured)
        if session:
            session.add_assistant(response)

        return {
            "response": response,
            "context": structured,
            "confidence": confidence,
            "clarify": False,
            "escalate": False,
            "grounding": grounding
        }

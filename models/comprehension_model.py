# Comprehension model implementation
"""
ComprehensionModel (updated)
- If a fine-tuned classifier checkpoint is available (COMP_CLASSIFIER_PATH or path in GENERATION_MODEL),
  it will load and use it for intent classification.
- Otherwise it falls back to a zero-shot NLI pipeline (configurable by COMPREHENSION_ZS_MODEL).
- Still performs rule-based entity extraction (symptoms, duration, severity).
"""
from typing import Dict, List, Optional
import os
import re
import logging
import json

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ComprehensionModel:
    def __init__(self, model_name: Optional[str] = None, classifier_path: Optional[str] = None, device: Optional[str] = None):
        """
        model_name: used for zero-shot fallback NLI model (e.g., facebook/bart-large-mnli)
        classifier_path: local path or HF repo id of a fine-tuned sequence classification model (optional)
        device: 'cpu' or 'cuda' (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_path = classifier_path or os.environ.get("COMP_CLASSIFIER_PATH")

        # if a classifier checkpoint is provided, load it
        if self.classifier_path and os.path.exists(self.classifier_path) or self.classifier_path and self.classifier_path.startswith("http"):
            try:
                logger.info(f"[ComprehensionModel] Loading classifier from {self.classifier_path} on {self.device}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.classifier_path)
                self.classifier_model = AutoModelForSequenceClassification.from_pretrained(self.classifier_path).to(self.device)
                # try to load label2id mapping from files in the checkpoint
                label_map_path = os.path.join(self.classifier_path, "label2id.json")
                if os.path.exists(label_map_path):
                    with open(label_map_path, "r") as f:
                        self.label2id = json.load(f)
                else:
                    # fallback to model config mapping if present
                    self.label2id = getattr(self.classifier_model.config, "label2id", None) or {}
                self.id2label = {v: k for k, v in self.label2id.items()} if self.label2id else getattr(self.classifier_model.config, "id2label", None) or {}
                self.use_classifier = True
            except Exception as e:
                logger.warning(f"[ComprehensionModel] Failed loading classifier {e}. Falling back to zero-shot.")
                self.use_classifier = False
                self.classifier_model = None
                self.tokenizer = None
        else:
            self.use_classifier = False
            self.classifier_model = None

        # zero-shot fallback
        if not self.use_classifier:
            self.zs_model = model_name or os.environ.get("COMPREHENSION_ZS_MODEL", "facebook/bart-large-mnli")
            logger.info(f"[ComprehensionModel] Using zero-shot model {self.zs_model}")
            self.zs_pipeline = pipeline("zero-shot-classification", model=self.zs_model, device=0 if self.device == "cuda" else -1)
            self.intent_labels = [
                "symptom_query",
                "general_question",
                "medication_query",
                "appointment_request",
                "other",
            ]

        # symptom lexicon and regexes (rule-based entity extraction)
        self.symptom_keywords = [
            "fever", "cough", "sore throat", "headache", "dizzy", "dizziness",
            "nausea", "vomit", "abdominal pain", "pain", "ache", "fatigue",
            "tired", "chills", "diarrhea", "shortness of breath", "breath",
            "rash", "itch", "congestion", "runny nose",
        ]
        pattern = r"\b(" + "|".join(re.escape(s) for s in self.symptom_keywords) + r")\b"
        self.symptom_re = re.compile(pattern, flags=re.IGNORECASE)
        self.severity_re = re.compile(r"\b(mild|moderate|severe|light|bad)\b", flags=re.IGNORECASE)
        self.duration_re = re.compile(
            r"(?:(?:for|since)\s+)?((?:\d+\s*(?:hours?|days?|weeks?|months?))|(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:hours?|days?|weeks?|months?)|yesterday|today)",
            flags=re.IGNORECASE,
        )

    def _classify_with_model(self, text: str) -> str:
        # tokenize and run the classifier_model
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        pred = int(np.argmax(logits).item())
        # map id to label if available
        if self.id2label:
            label = self.id2label.get(pred, str(pred))
        else:
            label = str(pred)
        return label

    def analyze_input(self, text: str) -> Dict:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        text_clean = text.strip()
        logger.info(f"[ComprehensionModel] Analyzing input: {text_clean}")

        # Intent detection
        if self.use_classifier and self.classifier_model is not None:
            try:
                intent = self._classify_with_model(text_clean)
            except Exception as e:
                logger.warning(f"[ComprehensionModel] classifier inference failed: {e}")
                intent = "other"
        else:
            try:
                zs_result = self.zs_pipeline(text_clean, candidate_labels=self.intent_labels, multi_label=False)
                intent = zs_result.get("labels", ["other"])[0]
            except Exception as e:
                logger.warning(f"[ComprehensionModel] zero-shot failed: {e}")
                intent = "other"

        # Symptom extraction
        symptoms = []
        for m in self.symptom_re.finditer(text_clean):
            symptom = m.group(0).lower()
            symptom = symptom.replace("vomiting", "vomit").strip()
            symptoms.append(symptom)
        seen = set()
        symptoms = [s for s in symptoms if not (s in seen or seen.add(s))]

        # Duration and severity
        duration_match = self.duration_re.search(text_clean)
        duration = duration_match.group(1) if duration_match else None
        sev_match = self.severity_re.search(text_clean)
        severity = sev_match.group(1).lower() if sev_match else None

        structured = {"intent": intent, "raw_text": text_clean, "symptoms": symptoms, "duration": duration, "severity": severity}
        logger.info(f"[ComprehensionModel] Structured output: {structured}")
        return structured

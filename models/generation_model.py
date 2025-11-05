# Generation model implementation
"""
GenerationModel (updated)
- Loads a seq2seq model from a HF id or a local checkpoint path (use GENERATION_MODEL env var or pass model_name).
- Safe generation with post-filtering and conservative fallback.
"""
from typing import Dict, Union, Optional
import os
import logging
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GenerationModel:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        # model_name can be an HF model id (t5-small, google/flan-t5-base) or a local path to a fine-tuned checkpoint
        self.model_name = model_name or os.environ.get("GENERATION_MODEL", "t5-small")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[GenerationModel] Loading {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.do_sample = os.environ.get("GEN_DO_SAMPLE", "false").lower() in ("1", "true", "yes")
        self.num_beams = int(os.environ.get("GEN_NUM_BEAMS", "4")) if not self.do_sample else 1
        self.max_length = int(os.environ.get("GEN_MAX_LENGTH", "180"))

        self.banned_prescription_re = re.compile(r"\b(take|prescribe|prescription|mg|tablet|pill|dose|dosage)\b", re.IGNORECASE)
        self.banned_diagnosis_re = re.compile(r"\b(diagnose|you have|you are suffering from|definitely|certainly)\b", re.IGNORECASE)

    def _build_prompt(self, context: Union[Dict, str]) -> str:
        if isinstance(context, dict):
            intent = context.get("intent", "unknown")
            raw = context.get("raw_text", "")
            symptoms = context.get("symptoms", [])
            duration = context.get("duration")
            severity = context.get("severity")
        else:
            intent = "unknown"
            raw = str(context)
            symptoms = []
            duration = None
            severity = None

        symptom_str = ", ".join(symptoms) if symptoms else "not specified"
        duration_str = duration if duration else "not specified"
        severity_str = severity if severity else "not specified"

        prompt = (
            "You are a cautious, helpful medical assistant. "
            "Do NOT give definitive diagnoses or specific prescriptions. "
            "Provide likely explanations, safe self-care suggestions (rest, hydration, symptomatic relief), "
            "and clear instructions to seek medical attention for red flags. Include a short disclaimer.\n\n"
            f"User intent: {intent}\n"
            f"Symptoms: {symptom_str}\n"
            f"Duration: {duration_str}\n"
            f"Severity: {severity_str}\n"
            f"User text: {raw}\n\n"
            "Answer concisely in 2-4 short sentences with a cautionary note."
        )
        return prompt

    def _safe_template(self, structured: Dict) -> str:
        parts = []
        if structured.get("symptoms"):
            parts.append(f"It sounds like you are experiencing {', '.join(structured['symptoms'])}.")
        else:
            parts.append("Thank you for the information.")
        parts.append("Try rest, hydration, and simple symptomatic measures (e.g., paracetamol for fever if appropriate).")
        if structured.get("duration"):
            parts.append(f"If symptoms persist beyond {structured['duration']} or worsen, seek medical care.")
        else:
            parts.append("If symptoms worsen or you develop concerning signs (difficulty breathing, high fever, fainting), seek urgent care.")
        parts.append("I am not a doctor. Consult a healthcare professional for diagnosis and treatment.")
        return " ".join(parts)

    def _post_filter(self, text: str, structured: Dict) -> str:
        if self.banned_prescription_re.search(text) or self.banned_diagnosis_re.search(text):
            logger.warning("[GenerationModel] Post-filter triggered. Replacing with safe template.")
            return self._safe_template(structured)
        return text

    def generate_response(self, context: Union[Dict, str]) -> str:
        structured = context if isinstance(context, dict) else {"raw_text": str(context)}
        prompt = self._build_prompt(structured)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams, "do_sample": self.do_sample, "early_stopping": True}
        if self.do_sample:
            gen_kwargs.update({"top_p": 0.95, "temperature": 0.7})

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except Exception as e:
            logger.exception(f"[GenerationModel] Generation failed: {e}")
            decoded = self._safe_template(structured)

        safe_text = self._post_filter(decoded, structured)
        return safe_text

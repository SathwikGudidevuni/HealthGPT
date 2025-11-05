# API implementation for HealthGPT
# app/api.py
"""
FastAPI application exposing the /ask endpoint that uses HealthPipeline.
Updated to accept optional session_id and to return clarification/escalation signals.
"""
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline import HealthPipeline

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

app = FastAPI(title="HealthGPT", version="0.3")

_pipeline: Optional[HealthPipeline] = None


class AskRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    response: str
    context: dict
    confidence: float
    clarify: bool
    escalate: bool
    grounding: list


@app.on_event("startup")
def startup_event():
    global _pipeline
    if _pipeline is not None:
        return
    comp = os.environ.get("COMPREHENSION_ZS_MODEL")
    gen = os.environ.get("GENERATION_MODEL")
    comp_classifier = os.environ.get("COMP_CLASSIFIER_PATH")
    comp_ner = os.environ.get("COMP_NER_PATH")
    retriever_path = os.environ.get("TRIAGE_RULES_PATH")
    device = None
    logger.info("[app.api] Starting HealthPipeline...")
    _pipeline = HealthPipeline(comp_model_name=comp, gen_model_name=gen, comp_classifier_path=comp_classifier, comp_ner_path=comp_ner, device=device, retriever_path=retriever_path)
    logger.info("[app.api] HealthPipeline ready.")


@app.get("/health")
def health():
    status = {"ready": _pipeline is not None}
    return status


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Request 'text' must be non-empty")
    try:
        result = _pipeline.run(req.text, session_id=req.session_id)
        return AskResponse(
            response=result["response"],
            context=result["context"],
            confidence=float(result.get("confidence", 0.0)),
            clarify=bool(result.get("clarify", False)),
            escalate=bool(result.get("escalate", False)),
            grounding=result.get("grounding", []),
        )
    except Exception:
        logger.exception("Error running pipeline")
        raise HTTPException(status_code=500, detail="Internal server error")

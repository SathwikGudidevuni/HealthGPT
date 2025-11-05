# Retrieval mechanism
# app/retrieval.py
"""
Simple retrieval/grounding helper.
For demo purposes this loads a small local triage rules JSON and returns
short grounding snippets relevant to detected symptoms or intent.

In production, replace with a vector store + FAISS / Weaviate / Pinecone search.
"""
import json
import os
from typing import List, Dict


DEFAULT_RULES_PATH = os.path.join("data", "triage_rules.json")


def load_rules(path: str = None) -> List[Dict]:
    p = path or DEFAULT_RULES_PATH
    if not os.path.exists(p):
        return []
    with open(p, "r") as f:
        return json.load(f)


class Retriever:
    def __init__(self, rules_path: str = None):
        self.rules = load_rules(rules_path)

    def retrieve(self, structured: Dict, top_k: int = 3) -> List[str]:
        """
        Return a list of short grounding strings relevant to symptoms/duration/severity.
        Uses simple keyword overlap for demo.
        """
        if not self.rules:
            return []

        text = structured.get("raw_text", "").lower()
        symptoms = structured.get("symptoms", [])
        matches = []
        for rule in self.rules:
            rule_text = rule.get("text", "")
            rule_keywords = rule.get("keywords", [])
            # match if any keyword present in text or symptoms
            if any(kw.lower() in text for kw in rule_keywords) or any(kw.lower() in s.lower() for kw in rule_keywords for s in symptoms):
                matches.append(rule_text)
        # dedupe and return top_k
        seen = set()
        out = []
        for m in matches:
            if m not in seen:
                out.append(m)
                seen.add(m)
            if len(out) >= top_k:
                break
        return out

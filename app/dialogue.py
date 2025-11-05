# Dialogue management
# app/dialogue.py
"""
Simple in-memory dialogue/session manager.
This is intentionally lightweight for demo use in Colab/local.
For production, replace with persistent store (Redis) and proper TTL/GC.
"""
import time
from typing import Dict, List, Optional


class SessionState:
    def __init__(self):
        # history: list of {"role": "user"|"assistant", "text": "...", "ts": unix_ts}
        self.history: List[Dict] = []
        self.meta: Dict = {}
        self.updated_at = time.time()

    def add_user(self, text: str):
        self.history.append({"role": "user", "text": text, "ts": time.time()})
        self.updated_at = time.time()

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "text": text, "ts": time.time()})
        self.updated_at = time.time()

    def summary(self, max_items: int = 6) -> str:
        # simple summary: return the last N turns joined
        last = self.history[-max_items:]
        return "\n".join(f"{h['role']}: {h['text']}" for h in last)


class DialogueManager:
    def __init__(self, session_ttl_seconds: int = 3600):
        self.sessions: Dict[str, SessionState] = {}
        self.session_ttl = session_ttl_seconds

    def get(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]

    def touch(self, session_id: str):
        s = self.get(session_id)
        s.updated_at = time.time()

    def prune(self):
        # remove expired sessions
        now = time.time()
        remove = [k for k, v in self.sessions.items() if (now - v.updated_at) > self.session_ttl]
        for k in remove:
            del self.sessions[k]

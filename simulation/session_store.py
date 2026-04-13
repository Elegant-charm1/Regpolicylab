from __future__ import annotations

import threading
import uuid
from typing import Dict, Optional

from .models import SimulationSession


class SimulationSessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SimulationSession] = {}
        self._latest_session_id: Optional[str] = None
        self._lock = threading.Lock()

    def create(self, model_type: str = "ollama") -> SimulationSession:
        session_id = uuid.uuid4().hex[:12]
        session = SimulationSession(session_id=session_id, model_type=model_type)
        self.save(session)
        return session

    def save(self, session: SimulationSession) -> SimulationSession:
        with self._lock:
            session.touch()
            self._sessions[session.session_id] = session
            self._latest_session_id = session.session_id
        return session

    def get(
        self, session_id: Optional[str] = None, allow_latest: bool = True
    ) -> Optional[SimulationSession]:
        with self._lock:
            if session_id:
                return self._sessions.get(session_id)
            if allow_latest and self._latest_session_id:
                return self._sessions.get(self._latest_session_id)
        return None

    def latest(self) -> Optional[SimulationSession]:
        return self.get(None, allow_latest=True)

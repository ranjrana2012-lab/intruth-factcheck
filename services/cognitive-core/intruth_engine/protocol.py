"""Shared WebSocket protocol — the lingua franca between all clients and the engine.

Two endpoints:
  /ws/audio  — clients (desktop capture, browser extension, phone) SEND audio here.
               First frame is a JSON header (text); subsequent frames are binary PCM.
  /ws/events — dashboard / clients RECEIVE transcripts, claims, verdicts here.

Keeping one schema across every platform is what makes the three front-ends thin and
interchangeable.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ─── Audio sources (declared by the client on connect) ──────────────────────
AudioSource = Literal["desktop_audio", "mic", "screen_ocr_text", "tab", "phone"]

SAMPLE_RATE = 16000  # 16 kHz mono — the wire format for ALL clients (matches Whisper)


class AudioConnectHeader(BaseModel):
    """Sent as the first (text) frame on /ws/audio before binary PCM begins."""

    source: AudioSource
    session_id: str
    consent: bool = True  # client asserts the user consented to capture
    sample_rate: int = SAMPLE_RATE
    language: str = "en"


# ─── Engine → subscribers event types ───────────────────────────────────────
class EventType(str, Enum):
    TRANSCRIPT = "transcript"
    PENDING_CLAIM = "pending_claim"
    VERDICT = "verdict"
    DROP_VERDICT = "drop_verdict"
    STATUS = "status"
    ERROR = "error"
    SPEAKER = "speaker"


class TranscriptEvent(BaseModel):
    """A finalized utterance (post-VAD, post-Whisper)."""

    type: Literal["transcript"] = "transcript"
    text: str
    speaker: str | None = None
    source: AudioSource | None = None
    ts: float = Field(default_factory=time.time)
    interim: bool = False


class ClaimEvent(BaseModel):
    """A check-worthy claim extracted from a transcript window (pending verdict)."""

    type: Literal["pending_claim"] = "pending_claim"
    claim: str
    speaker: str | None = None
    source: AudioSource | None = None
    ts: float = Field(default_factory=time.time)


class Evidence(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    date: str = ""
    kind: Literal["direct_answer", "knowledge_panel", "organic"] = "organic"


class VerdictLabel(str, Enum):
    TRUE = "TRUE"
    SUBSTANTIALLY_TRUE = "SUBSTANTIALLY TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNVERIFIABLE = "UNVERIFIABLE"


class VerdictEvent(BaseModel):
    """A final verdict with evidence and source-bias annotations."""

    type: Literal["verdict"] = "verdict"
    claim: str
    verdict: str  # VerdictLabel value
    confidence: Literal["HIGH", "LOW"] = "HIGH"
    explanation: str = ""
    speaker: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    bias: list[dict[str, Any]] = Field(default_factory=list)  # [{domain, bias, credibility}]
    lexical: dict[str, Any] | None = None
    ts: float = Field(default_factory=time.time)


class DropVerdictEvent(BaseModel):
    """Tell subscribers to remove a pending card (inversion detected / unverifiable)."""

    type: Literal["drop_verdict"] = "drop_verdict"
    claim: str


class StatusEvent(BaseModel):
    """Engine lifecycle / capture-state changes (drives the tray indicator)."""

    type: Literal["status"] = "status"
    capturing: bool
    sources: list[AudioSource] = Field(default_factory=list)
    message: str = ""


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


# Union for easy (de)serialization
Event = (
    TranscriptEvent
    | ClaimEvent
    | VerdictEvent
    | DropVerdictEvent
    | StatusEvent
    | ErrorEvent
)


def dumps(event: BaseModel) -> str:
    """Serialize an event to a JSON string for the wire."""
    return event.model_dump_json()


# ─── Binary PCM helpers (shared with clients) ───────────────────────────────
def pcm_float32_to_int16(float32: bytes) -> bytes:
    """Convert raw Float32 PCM bytes → Int16 PCM bytes (the Whisper/VAD input format).

    Mirrors the original InTruth offscreen.js conversion:
        int16[i] = clamp(float32[i] * 32768, -32768, 32767)
    """
    import struct

    n = len(float32) // 4
    floats = struct.unpack(f"<{n}f", float32[: n * 4])
    ints = [max(-32768, min(32767, int(f * 32768))) for f in floats]
    return struct.pack(f"<{n}h", *ints)

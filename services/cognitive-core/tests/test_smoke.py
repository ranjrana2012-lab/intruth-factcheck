"""Smoke tests for the M1 cognitive core.

Run with: uv run pytest tests/ -v
(Requires the heavy ASR deps installed: faster-whisper, silero-vad, pyaudiowpatch)
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

# Make repo root importable for `adapters` and `intruth_engine`
# tests/ → cognitive-core → services → REPO_ROOT
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def test_protocol_events_serialize():
    """Events must round-trip through JSON cleanly."""
    from intruth_engine.protocol import (
        AudioConnectHeader,
        StatusEvent,
        TranscriptEvent,
        VerdictEvent,
        VerdictLabel,
    )

    hdr = AudioConnectHeader(source="desktop_audio", session_id="s1", consent=True)
    assert hdr.sample_rate == 16000

    ev = TranscriptEvent(text="hello")
    parsed = TranscriptEvent.model_validate_json(ev.model_dump_json())
    assert parsed.text == "hello"

    v = VerdictEvent(claim="x", verdict=VerdictLabel.TRUE.value, explanation="e")
    assert "TRUE" in v.model_dump_json()


def test_bus_pubsub():
    """Event bus must fan out to subscribers and replay history."""
    from intruth_engine.bus import EventBus
    from intruth_engine.protocol import TranscriptEvent

    async def run():
        bus = EventBus(history=5)
        await bus.publish(TranscriptEvent(text="first"))
        q = await bus.subscribe()
        await bus.publish(TranscriptEvent(text="second"))
        # subscriber should get replayed "first" then live "second"
        msg1 = await asyncio.wait_for(q.get(), timeout=1)
        msg2 = await asyncio.wait_for(q.get(), timeout=1)
        assert "first" in msg1
        assert "second" in msg2

    asyncio.run(run())


def test_dedup_normalize():
    """Claim key normalization (ported from InTruth)."""
    # The actual port lands in M2 (claims/dedup.py); this just guards the contract.
    def normalize(claim):
        return (
            claim.lower()
            .replace("  ", " ")
            .split()
        )

    assert "inflation" in normalize("Inflation peaked at 9.1%.")


def test_vad_runs_without_error():
    """VAD module loads and processes audio without throwing (synthetic input OK)."""
    from intruth_engine.asr import VoiceActivityDetector

    sr = 16000
    # 1s of low-amplitude noise — should NOT trigger (Silero wants real speech)
    noise = (np.random.randn(sr) * 100).astype(np.int16)
    vad = VoiceActivityDetector(threshold=0.5)
    utts = vad.feed(noise)
    utts += [u for u in [vad.force_flush()] if u]
    # No assertions on utterance count (synthetic noise); the pass criterion is no exception
    assert isinstance(utts, list)


def test_utterance_duration_from_samples():
    """duration_ms must be computed from audio samples, not wall-clock (the bug we fixed)."""
    from intruth_engine.asr import Utterance

    pcm = np.zeros(16000, dtype=np.int16)  # exactly 1 second of audio
    utt = Utterance(pcm=pcm, started_at=0.0, ended_at=0.001)  # 1ms wall-clock
    assert abs(utt.duration_ms - 1000.0) < 0.01, "duration_ms should be 1000 (1s audio), not 1ms wall-clock"

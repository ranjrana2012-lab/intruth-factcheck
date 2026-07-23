"""faster-whisper wrapper — local streaming-style transcription.

Auto-detects CUDA (your GTX 1660 SUPER) and falls back to CPU. faster-whisper (CTranslate2)
is up to 4× faster than openai/whisper and supports INT8 quantization for CPU. We only
transcribe complete utterances handed to us by the VAD — never raw silence.

Memory note: the model is loaded once at startup and reused. Utterance audio is transcribed
then dropped (never persisted).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..config import asr_config, resolve_device
from .vad import Utterance

log = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start: float  # seconds from utterance start
    end: float
    utterance: Utterance  # carry through for timestamps + source attribution


class WhisperTranscriber:
    """Singleton transcriber. Thread-safe-ish for sequential asyncio-driven calls."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        cfg = asr_config()
        self.model_name = model_name or cfg.model
        self._requested_device = device or cfg.device
        self._model = None  # lazy

    def _resolve_compute_type(self, device: str) -> str:
        cfg = asr_config()
        if cfg.compute_type != "auto":
            return cfg.compute_type
        # float16 on CUDA (fast, fits in 6GB for small/medium models); int8 on CPU.
        return "float16" if device == "cuda" else "int8"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        device = resolve_device(self._requested_device)
        compute_type = self._resolve_compute_type(device)
        from faster_whisper import WhisperModel

        log.info("loading faster-whisper '%s' on %s (%s)…", self.model_name, device, compute_type)
        self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
        log.info("faster-whisper ready (device=%s)", device)

    def transcribe(self, utterance: Utterance, language: str | None = None) -> list[TranscriptSegment]:
        """Transcribe one utterance → segments. Blocking; run in executor."""
        self._ensure_model()
        audio = utterance.pcm.astype("float32") / 32768.0  # faster-whisper wants float32 [-1,1]
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=False,  # we already VAD'd upstream
            without_timestamps=False,
        )
        result = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                result.append(
                    TranscriptSegment(
                        text=text,
                        start=seg.start,
                        end=seg.end,
                        utterance=utterance,
                    )
                )
        return result


# Module-level singleton (one model load per process)
_transcriber: WhisperTranscriber | None = None


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


async def asr_task(
    utterance_queue: "asyncio.Queue[Utterance]",
    on_transcript,  # callable: async (TranscriptSegment) -> None
    language: str = "en",
) -> None:
    """Async task: drain VAD utterances → transcribe → emit transcript segments."""
    transcriber = get_transcriber()
    loop = asyncio.get_running_loop()
    log.info("asr task started (language=%s)", language)
    while True:
        utt = await utterance_queue.get()
        if utt is None:  # shutdown sentinel
            break
        try:
            segments = await loop.run_in_executor(None, transcriber.transcribe, utt, language)
            for seg in segments:
                await on_transcript(seg)
        except Exception:
            log.exception("asr: transcription failed for a %.1fs utterance", utt.duration_ms / 1000)
    log.info("asr task stopped")

"""Silero VAD — voice activity detection that gates transcription.

Why: transcribing silence/noise with Whisper burns CPU/GPU for nothing. Silero runs on a
single CPU thread at <1ms per 30ms chunk, so it's the cheap gatekeeper in front of the
expensive STT. This is the pattern every ambient-listening assistant uses (Granola, tl;dv).

Pipeline: continuous Int16 PCM in → 30ms frames scored by Silero → speech buffered → on
≥MIN_SILENCE_MS of silence, flush the buffered speech as one utterance chunk.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Silero operates on 30ms frames at 16kHz → 512 samples per frame for this silero build.
SAMPLE_RATE = 16000
FRAME_SAMPLES_16K = 512  # silero-vad v6 expects 512-sample frames at 16kHz
DEFAULT_THRESHOLD = 0.5
MIN_SILENCE_MS = 700
MIN_SPEECH_MS = 300


@dataclass
class Utterance:
    """A buffered speech segment ready for transcription."""

    pcm: np.ndarray  # Int16, mono, 16kHz
    started_at: float  # wall-clock time speech began (for logging/UI)
    ended_at: float  # wall-clock time speech ended
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_ms(self) -> float:
        """Audio duration (from sample count), NOT wall-clock time.

        Wall-clock would under-report when processing is faster than realtime (GPU), which
        incorrectly drops valid utterances in the min_speech_ms filter.
        """
        return len(self.pcm) / self.sample_rate * 1000


@dataclass
class VoiceActivityDetector:
    """Wraps Silero VAD into a streaming-friendly chunker.

    Feed Int16 PCM via `feed()`; get complete Utterances back. Uses Silero's VADIterator
    which emits {start, end} boundaries as speech begins/ends. We buffer the raw frames
    while inside a speech region and emit one Utterance per region.
    """

    threshold: float = DEFAULT_THRESHOLD
    min_silence_ms: int = MIN_SILENCE_MS
    min_speech_ms: int = MIN_SPEECH_MS
    _model: object | None = field(default=None, repr=False)
    _iterator: object | None = field(default=None, repr=False)
    _speech_buffer: list[np.ndarray] = field(default_factory=list, repr=False)
    _speaking: bool = False
    _speech_started_at: float | None = None
    _carryover: np.ndarray | None = field(default=None, repr=False)  # incomplete frame bytes

    def _ensure_model(self) -> None:
        if self._iterator is not None:
            return
        try:
            from silero_vad import VADIterator, load_silero_vad

            self._model = load_silero_vad(onnx=True)
            self._iterator = VADIterator(
                self._model,
                threshold=self.threshold,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=80,
            )
            log.info("silero-vad loaded (onnx, threshold=%.2f)", self.threshold)
        except Exception as e:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "Failed to load silero-vad. Install with: pip install silero-vad onnxruntime"
            ) from e

    def feed(self, pcm_int16: np.ndarray) -> list[Utterance]:
        """Feed a chunk of Int16 PCM; return any utterances that completed.

        Chunks may be any size; we internally re-chunk into 512-sample frames, carrying
        any remainder to the next call.
        """
        self._ensure_model()
        assert self._iterator is not None

        # Prepend any carried-over samples from the last call
        if self._carryover is not None and len(self._carryover):
            pcm_int16 = np.concatenate([self._carryover, pcm_int16])
        n_frames = len(pcm_int16) // FRAME_SAMPLES_16K
        # Carry the remainder for next time
        remainder = len(pcm_int16) - (n_frames * FRAME_SAMPLES_16K)
        if remainder:
            self._carryover = pcm_int16[n_frames * FRAME_SAMPLES_16K :].copy()
        else:
            self._carryover = None

        results: list[Utterance] = []
        for i in range(n_frames):
            start = i * FRAME_SAMPLES_16K
            frame_raw = pcm_int16[start : start + FRAME_SAMPLES_16K]
            # Silero expects float32 in [-1, 1]
            frame_f32 = frame_raw.astype(np.float32) / 32768.0
            t = time.time()
            speech_dict = self._iterator(frame_f32, return_seconds=False)

            if speech_dict:
                # Speech start boundary
                if "start" in speech_dict and not self._speaking:
                    self._speaking = True
                    self._speech_started_at = t
                    self._speech_buffer = []
                    log.debug("vad: speech start @ %.2f", t)
                # Speech end boundary
                if "end" in speech_dict and self._speaking:
                    utt = self._flush(t)
                    if utt:
                        results.append(utt)

            # Buffer raw frames while inside a speech region (VADIterator only reports
            # start/end transitions, so we accumulate continuously between them)
            if self._speaking:
                self._speech_buffer.append(frame_raw)

        return results

    def _flush(self, now: float) -> Utterance | None:
        if not self._speech_buffer:
            self._speaking = False
            return None
        pcm = np.concatenate(self._speech_buffer)
        started = self._speech_started_at or now
        utt = Utterance(pcm=pcm, started_at=started, ended_at=now)
        self._speech_buffer = []
        self._speaking = False
        self._speech_started_at = None
        if utt.duration_ms < self.min_speech_ms:
            log.debug("vad: dropping %.0fms segment (too short)", utt.duration_ms)
            return None
        log.debug("vad: flushed %.1fs utterance", utt.duration_ms / 1000)
        return utt

    def force_flush(self) -> Utterance | None:
        """Flush any in-progress speech (e.g. on capture stop)."""
        if self._speaking:
            return self._flush(time.time())
        return None


async def vad_task(
    audio_queue: "asyncio.Queue[np.ndarray]",
    utterance_queue: "asyncio.Queue[Utterance]",
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """Async task: drain raw PCM → VAD → emit complete utterances.

    Runs the blocking Silero inference in a worker thread so it can't stall the loop.
    """
    vad = VoiceActivityDetector(threshold=threshold)
    loop = asyncio.get_running_loop()
    log.info("vad task started (threshold=%.2f)", threshold)
    while True:
        pcm = await audio_queue.get()
        if pcm is None:  # shutdown sentinel
            utt = vad.force_flush()
            if utt:
                await utterance_queue.put(utt)
            break
        utterances = await loop.run_in_executor(None, vad.feed, pcm)
        for utt in utterances:
            await utterance_queue.put(utt)
    log.info("vad task stopped")

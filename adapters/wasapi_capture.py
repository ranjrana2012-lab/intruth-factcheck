"""WASAPI loopback audio capture for Windows — captures whatever the system is playing.

Uses PyAudioWPatch (a PyAudio fork with WASAPI loopback support). This is the same approach
Screenpipe uses internally. Loopback captures the output mix — so podcasts, YouTube, calls,
and any audio your speakers/headphones produce are all captured.

Device-switch handling: if the user changes their default audio output mid-stream, the
device info goes stale and we re-open the stream on the new default output.

NOTE: For non-Windows, fall back to standard PyAudio input (mic). For full cross-platform
system-audio loopback, Screenpipe is the production path (M1 production swap).
"""
from __future__ import annotations

import asyncio
import logging
import platform
import time

import numpy as np

from intruth_engine.config import audio_config

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_FRAMES = 1024  # frames per PyAudio read


class WasapiCapture:
    """WASAPI loopback (Windows) / mic (other OS). Yields Int16 16kHz mono PCM."""

    def __init__(self, capture_system_audio: bool = True, capture_mic: bool = False):
        self.capture_system_audio = capture_system_audio and platform.system() == "Windows"
        self.capture_mic = capture_mic
        self._pyaudio = None
        self._stream = None
        self._running = False

    def _open_pyaudio(self):
        if self._pyaudio is not None:
            return
        try:
            import pyaudiowpatch as pyaudio  # PyAudioWPatch (Windows loopback)
        except ImportError:
            try:
                import pyaudio as pyaudio  # plain PyAudio fallback (mic only)
                self.capture_system_audio = False
                log.warning("pyaudiowpatch not found; falling back to plain pyaudio (mic only)")
            except ImportError as e:
                raise RuntimeError(
                    "Neither pyaudiowpatch nor pyaudio installed. "
                    "Install with: pip install pyaudiowpatch"
                ) from e
        self._pyaudio = pyaudio.PyAudio()

    def _open_stream(self):
        """Open a loopback stream on the default output device (Windows), else mic."""
        pa = self._pyaudio
        if self.capture_system_audio:
            try:
                default_speaker = pa.get_default_wasapi_loopback()
            except OSError:
                log.warning("WASAPI loopback unavailable; falling back to mic input")
                default_speaker = None
            if default_speaker:
                log.info(
                    "WASAPI loopback: %s (%d Hz)",
                    default_speaker["name"],
                    default_speaker["defaultSampleRate"],
                )
                return pa.open(
                    format=pa.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=default_speaker["index"],
                    frames_per_buffer=CHUNK_FRAMES,
                )
        # Mic fallback (any OS)
        log.info("opening default mic input (%d Hz)", SAMPLE_RATE)
        return pa.open(
            format=pa.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )

    def _read_chunk(self) -> np.ndarray | None:
        """Read one PCM chunk. Returns Int16 numpy array or None on error."""
        try:
            raw = self._stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            return np.frombuffer(raw, dtype=np.int16)
        except OSError as e:
            log.warning("audio read error (will reopen): %s", e)
            return None

    async def stream(self, out_queue: "asyncio.Queue[np.ndarray | None]"):
        """Capture loop: read PCM → push to queue. Runs until cancelled or stop sentinel."""
        self._open_pyaudio()
        self._stream = self._open_stream()
        self._running = True
        log.info("audio capture started")
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                chunk = await loop.run_in_executor(None, self._read_chunk)
                if chunk is None:
                    # device hiccup — reopen
                    try:
                        self._stream.stop_stream()
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = self._open_stream()
                    continue
                await out_queue.put(chunk)
        finally:
            await out_queue.put(None)  # shutdown sentinel downstream
            self.close()

    def stop(self):
        self._running = False

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
        log.info("audio capture stopped")


def make_default_capture() -> WasapiCapture:
    """Build a capture from config: WASAPI loopback + optional mic."""
    cfg = audio_config()
    return WasapiCapture(
        capture_system_audio=cfg.enabled and cfg.adapter in ("wasapi_loopback", "screenpipe"),
        capture_mic=cfg.microphone_enabled,
    )

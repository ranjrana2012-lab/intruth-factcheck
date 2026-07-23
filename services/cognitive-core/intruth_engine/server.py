"""FastAPI cognitive-core server.

Two WebSocket endpoints + a dashboard:
  /ws/audio  — clients SEND audio (JSON header frame, then binary PCM) or receive our
               own desktop capture when `autocapture=1`.
  /ws/events — any client (dashboard, extension, phone) SUBSCRIBES to transcripts/verdicts.

The /ws/audio path also lets the engine run its OWN native capture (WASAPI loopback) in
a background task, so the desktop doesn't need a separate client app for M1 — just open the
dashboard and it works.

M1 scope: capture → VAD → Whisper → transcript events. Claim extraction + verification
(M2) plugs into `on_transcript` below.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .asr import TranscriptSegment, asr_task, vad_task
from .bus import bus, publish_status
from .config import audio_config, get_settings
from .protocol import (
    SAMPLE_RATE,
    AudioConnectHeader,
    TranscriptEvent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("intruth.engine")

DASHBOARD_DIR = Path(__file__).parent / "dashboard"


# ─── Engine state ────────────────────────────────────────────────────────────
class EngineState:
    """Holds the asyncio queues + capture handle for the engine's own desktop capture."""

    def __init__(self):
        self.audio_queue: asyncio.Queue | None = None  # raw PCM → VAD
        self.utterance_queue: asyncio.Queue | None = None  # VAD utterances → ASR
        self.capture = None  # WasapiCapture | None
        self.tasks: list[asyncio.Task] = []
        self.active_clients: int = 0  # external audio-sending clients
        self.sources: set[str] = set()
        self.capturing: bool = False


state = EngineState()


async def on_transcript(seg: TranscriptSegment, source: str = "desktop_audio") -> None:
    """Callback when Whisper emits a finalized segment.

    In M1: publish a transcript event. In M2: this is also where Presidio → claim
    windowing → dedup → verify pipeline hooks in.
    """
    event = TranscriptEvent(text=seg.text, source=source)  # type: ignore[arg-type]
    log.info("transcript [%s]: %s", source, seg.text)
    await bus.publish(event)
    # M2 hook: await claims_pipeline.on_transcript(seg, source)


# ─── Desktop capture lifecycle (engine's own WASAPI loopback) ────────────────
async def start_desktop_capture() -> None:
    """Start the engine's native desktop capture + VAD + ASR pipeline."""
    if state.capturing:
        return
    # Import here so the server module imports cleanly on machines without pyaudio
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from adapters import make_default_capture

    state.audio_queue = asyncio.Queue(maxsize=1000)
    state.utterance_queue = asyncio.Queue(maxsize=200)
    state.capture = make_default_capture()

    cfg = audio_config()
    state.tasks = [
        asyncio.create_task(state.capture.stream(state.audio_queue), name="capture"),
        asyncio.create_task(
            vad_task(state.audio_queue, state.utterance_queue, threshold=cfg.vad_sensitivity),
            name="vad",
        ),
        asyncio.create_task(
            asr_task(
                state.utterance_queue,
                on_transcript=lambda seg: on_transcript(seg, "desktop_audio"),
                language="en",
            ),
            name="asr",
        ),
    ]
    state.capturing = True
    state.sources.add("desktop_audio")
    await publish_status(True, list(state.sources), "Desktop capture started")
    log.info("desktop capture pipeline started (sources=%s)", state.sources)


async def stop_desktop_capture() -> None:
    """Gracefully stop the engine's capture pipeline."""
    if not state.capturing:
        return
    if state.capture:
        state.capture.stop()
    # Sendin shutdown sentinels down the queues
    if state.audio_queue:
        await state.audio_queue.put(None)
    for t in state.tasks:
        t.cancel()
    try:
        await asyncio.gather(*state.tasks, return_exceptions=True)
    except Exception:
        pass
    state.tasks = []
    state.capture = None
    state.capturing = False
    state.sources.discard("desktop_audio")
    await publish_status(False, list(state.sources), "Desktop capture stopped")
    log.info("desktop capture pipeline stopped")


# ─── FastAPI app ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("InTruth engine starting on %s:%s", settings.engine_host, settings.engine_port)
    yield
    await stop_desktop_capture()
    log.info("InTruth engine stopped")


app = FastAPI(title="InTruth FactCheck Engine", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def index():
    """Serve the dashboard (or a pointer to it)."""
    index_html = DASHBOARD_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(index_html.read_text(encoding="utf-8"))
    return JSONResponse(
        {"service": "intruth-engine", "status": "ok", "dashboard": "not yet built"},
        status_code=200,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "capturing": state.capturing,
        "sources": list(state.sources),
        "subscribers": bus.subscriber_count(),
    }


@app.post("/capture/{action}")
async def capture_control(action: str):
    """Start/stop the engine's own desktop capture (also drivable from the tray)."""
    if action == "start":
        await start_desktop_capture()
        return {"ok": True, "capturing": state.capturing}
    if action == "stop":
        await stop_desktop_capture()
        return {"ok": True, "capturing": state.capturing}
    return JSONResponse({"error": "use /capture/start or /capture/stop"}, status_code=400)


# ─── /ws/audio — clients send PCM, OR request the engine's own capture ──────
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    """External client streaming audio into the engine.

    Frame 1 (text): AudioConnectHeader JSON. Subsequent frames: binary Int16 PCM.
    (In M3, the browser extension and phone companion stream here.)
    """
    await ws.accept()
    try:
        header_raw = await ws.receive_text()
        header = AudioConnectHeader.model_validate_json(header_raw)
        if not header.consent:
            await ws.close(code=4003, reason="consent required")
            return

        # Per-client pipeline: VAD + ASR fed from this socket's PCM
        audio_q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        utt_q: asyncio.Queue = asyncio.Queue(maxsize=200)
        cfg = audio_config()
        tasks = [
            asyncio.create_task(vad_task(audio_q, utt_q, threshold=cfg.vad_sensitivity)),
            asyncio.create_task(
                asr_task(utt_q, lambda seg: on_transcript(seg, header.source), language=header.language)
            ),
        ]
        state.active_clients += 1
        state.sources.add(header.source)
        await publish_status(True, list(state.sources))
        log.info("audio client connected: source=%s session=%s", header.source, header.session_id)

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    pcm = np.frombuffer(msg["bytes"], dtype=np.int16)
                    try:
                        audio_q.put_nowait(pcm)
                    except asyncio.QueueFull:
                        pass  # drop under load rather than stall the socket
        finally:
            await audio_q.put(None)  # shutdown sentinel
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            state.active_clients -= 1
            if state.active_clients == 0:
                state.sources.discard(header.source)
            await publish_status(state.capturing or state.active_clients > 0, list(state.sources))
            log.info("audio client disconnected: source=%s", header.source)
    except WebSocketDisconnect:
        log.info("audio websocket disconnected")
    except Exception:
        log.exception("audio websocket error")


# ─── /ws/events — subscribe to transcripts / verdicts ───────────────────────
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """Any client subscribes here to receive the live event stream (dashboard, phone, ext)."""
    await ws.accept()
    q = await bus.subscribe()
    log.info("event subscriber connected (total=%d)", bus.subscriber_count())
    try:
        while True:
            serialized = await q.get()
            await ws.send_text(serialized)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("event websocket error")
    finally:
        await bus.unsubscribe(q)
        log.info("event subscriber disconnected (total=%d)", bus.subscriber_count())


def main():
    """Entry point: `python -m intruth_engine.server` or `intruth-engine`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "intruth_engine.server:app",
        host=settings.engine_host,
        port=settings.engine_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

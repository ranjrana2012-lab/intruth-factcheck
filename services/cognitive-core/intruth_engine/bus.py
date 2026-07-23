"""In-process async event bus: fan-out of engine events to all /ws/events subscribers.

One bus instance (module-level singleton). The ASR/claim/verify pipeline publishes here;
WebSocket subscribers (dashboard, extension, phone) consume. Decouples producers from
consumers so the pipeline never blocks on a slow client.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING

from .protocol import StatusEvent, dumps

if TYPE_CHECKING:
    from .protocol import Event  # noqa: F401

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, history: int = 50):
        # Each subscriber gets its own asyncio.Queue. New subscribers also receive
        # the last `history` events so a freshly-opened dashboard isn't blank.
        self._subscribers: set[asyncio.Queue] = set()
        self._recent: deque[str] = deque(maxlen=history)
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        async with self._lock:
            self._subscribers.add(q)
            for past in self._recent:
                await q.put(past)  # replay recent history
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, event: "Event") -> None:
        """Publish an event to all subscribers (non-blocking, drops on overflow)."""
        serialized = dumps(event)
        async with self._lock:
            self._recent.append(serialized)
            dead: list[asyncio.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(serialized)
                except asyncio.QueueFull:
                    # slow subscriber — drop oldest, push newest (keep it live)
                    try:
                        q.get_nowait()
                        q.put_nowait(serialized)
                    except Exception:
                        dead.append(q)
            for q in dead:
                self._subscribers.discard(q)

    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Module-level singleton
bus = EventBus()


async def publish_status(capturing: bool, sources: list[str] | None = None, message: str = "") -> None:
    await bus.publish(StatusEvent(capturing=capturing, sources=sources or [], message=message))

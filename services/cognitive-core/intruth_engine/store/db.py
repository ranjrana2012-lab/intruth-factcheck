"""SQLite store — persists claims + verdicts ONLY (never raw audio, never full transcripts).

This is the privacy posture: we keep the fact-check output (useful, reviewable) but not the
raw capture that produced it. Uses aiosqlite for async access from the pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from ..config import data_dir

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence TEXT,
    explanation TEXT,
    speaker TEXT,
    sources_json TEXT,        -- JSON array of URLs
    bias_json TEXT,           -- JSON array of bias pills
    source TEXT,              -- desktop_audio | mic | phone | ...
    language TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verdicts_created ON verdicts(created_at);
CREATE INDEX IF NOT EXISTS idx_verdicts_claim ON verdicts(claim);
"""


async def init_db(db_path=None) -> str:
    """Create the DB + schema. Returns the path."""
    import aiosqlite

    db_path = db_path or str(data_dir() / "intruth.db")
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_SCHEMA)
        await db.commit()
    log.info("sqlite store ready: %s", db_path)
    return db_path


async def store_verdict(
    claim: str,
    verdict: str,
    confidence: str = "HIGH",
    explanation: str = "",
    speaker: str | None = None,
    sources: list[str] | None = None,
    bias: list[dict] | None = None,
    source: str = "desktop_audio",
    language: str = "en",
    db_path: str | None = None,
) -> int:
    """Persist one verdict. Returns the row id."""
    import aiosqlite

    db_path = db_path or str(data_dir() / "intruth.db")
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            """INSERT INTO verdicts
               (claim, verdict, confidence, explanation, speaker, sources_json, bias_json, source, language, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim, verdict, confidence, explanation, speaker,
                json.dumps(sources or []), json.dumps(bias or []),
                source, language, datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()
        return cur.lastrowid or 0


async def recent_verdicts(limit: int = 50, db_path: str | None = None) -> list[dict]:
    import aiosqlite

    db_path = db_path or str(data_dir() / "intruth.db")
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM verdicts ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [
            {
                **dict(r),
                "sources": json.loads(r["sources_json"] or "[]"),
                "bias": json.loads(r["bias_json"] or "[]"),
            }
            for r in rows
        ]

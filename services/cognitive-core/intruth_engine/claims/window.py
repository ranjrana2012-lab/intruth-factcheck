"""Sentence windowing — ported from InTruth's onNewSentence / rolling window.

Accumulates finalized transcript sentences into a rolling buffer and fires evaluation
when the window is full (WINDOW_SIZE sentences) OR on a speaker change mid-window.
This batches context so claim-extraction sees several sentences at once (better accuracy
than per-sentence), while keeping latency bounded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .lexical import LexicalFeatures, build_lexical_summary, extract_lexical


@dataclass
class WindowEntry:
    text: str
    speaker_id: int | None = None
    speaker_name: str | None = None


@dataclass
class WindowSnapshot:
    """Emitted when the window fills or a speaker change flushes it."""

    context_text: str  # joined sentences, with optional [Speaker] labels
    dominant_speaker: str | None
    dominant_speaker_id: int | None
    opponent_name: str | None
    lexical_summary: str
    lexical: LexicalFeatures


@dataclass
class SentenceWindow:
    """Rolling sentence window. Feed finalized sentences; collect WindowSnapshots.

    Port of InTruth's onNewSentence. Two fire conditions:
      - window full (sentence_count % WINDOW_SIZE == 0)
      - speaker change mid-window (with ≥2 sentences buffered) — flushes early
    """

    window_size: int = 4
    window_keep: int = 15
    _buffer: list[WindowEntry] = field(default_factory=list)
    _count: int = 0
    _lexical_acc: LexicalFeatures = field(default_factory=LexicalFeatures.neutral)
    _window_start: float | None = None
    _last_speaker_id: int | None = None
    _speaker_names: dict[int, str] = field(default_factory=dict)  # confirmed id→name

    def set_speaker_name(self, speaker_id: int, name: str) -> None:
        self._speaker_names[speaker_id] = name

    def _dominant(self) -> tuple[int | None, str | None]:
        """Most frequent speaker in the current window slice."""
        current = self._buffer[-self.window_size:]
        counts: dict[int, int] = {}
        for e in current:
            if e.speaker_id is not None:
                counts[e.speaker_id] = counts.get(e.speaker_id, 0) + 1
        if not counts:
            return None, None
        dom_id = max(counts, key=counts.get)
        return dom_id, self._speaker_names.get(dom_id)

    def _opponent(self, dominant_name: str | None) -> str | None:
        names = list(self._speaker_names.values())
        if len(names) >= 2 and dominant_name:
            for n in names:
                if n.lower() != dominant_name.lower():
                    return n
        return None

    def _snapshot(self) -> WindowSnapshot:
        dom_id, dom_name = self._dominant()
        # Average accumulated lexical rates over the window's sentence count
        sc = max(self._lexical_acc.rates.get("_sentenceCount", 0) if False else 1, 1)
        rates = {k: round(v / sc) if k != "_sentenceCount" else v for k, v in self._lexical_acc.rates.items()}
        # speech rate
        if self._window_start:
            elapsed = time.time() - self._window_start
            if elapsed > 0 and self._lexical_acc.word_count:
                rates_wpsec = round(self._lexical_acc.word_count / elapsed, 1)
                lex = LexicalFeatures(rates=rates, words_per_second=rates_wpsec, word_count=self._lexical_acc.word_count)
            else:
                lex = LexicalFeatures(rates=rates, word_count=self._lexical_acc.word_count)
        else:
            lex = LexicalFeatures(rates=rates, word_count=self._lexical_acc.word_count)
        # build context with optional speaker labels
        parts = []
        for e in self._buffer[-self.window_size:]:
            if e.speaker_name:
                parts.append(f"[{e.speaker_name}] {e.text}")
            elif e.speaker_id is not None:
                parts.append(f"[Speaker {e.speaker_id}] {e.text}")
            else:
                parts.append(e.text)
        return WindowSnapshot(
            context_text=" ".join(parts),
            dominant_speaker=dom_name,
            dominant_speaker_id=dom_id,
            opponent_name=self._opponent(dom_name),
            lexical_summary=build_lexical_summary(lex),
            lexical=lex,
        )

    def _reset_for_next_window(self) -> None:
        self._lexical_acc = LexicalFeatures.neutral()
        self._window_start = None

    def feed(self, text: str, speaker_id: int | None = None) -> list[WindowSnapshot]:
        """Add a finalized sentence. Returns any snapshots that fired."""
        snapshots: list[WindowSnapshot] = []
        speaker_name = self._speaker_names.get(speaker_id) if speaker_id is not None else None

        # Speaker-change flush (mid-window, ≥2 buffered, not already at a window boundary)
        if (
            self._last_speaker_id is not None
            and speaker_id is not None
            and speaker_id != self._last_speaker_id
            and self._count % self.window_size != 0
            and len(self._buffer) >= 2
        ):
            snapshots.append(self._snapshot())
            self._reset_for_next_window()
        self._last_speaker_id = speaker_id

        # Append + accumulate lexical
        self._buffer.append(WindowEntry(text=text, speaker_id=speaker_id, speaker_name=speaker_name))
        if len(self._buffer) > self.window_keep:
            self._buffer.pop(0)
        self._count += 1
        if self._window_start is None:
            self._window_start = time.time()
        feat = extract_lexical(text)
        for k in ("hedging", "certainty", "filler", "emotional", "exclusive", "firstPersonSg"):
            self._lexical_acc.rates[k] = self._lexical_acc.rates.get(k, 0) + feat.rates.get(k, 0)
        self._lexical_acc.word_count += feat.word_count
        self._lexical_acc.rates["_sentenceCount"] = self._lexical_acc.rates.get("_sentenceCount", 0) + 1

        # Window-full fire
        if self._count % self.window_size == 0:
            snapshots.append(self._snapshot())
            self._reset_for_next_window()

        return snapshots

    def flush(self) -> list[WindowSnapshot]:
        """Force-emit any pending buffer (e.g. on capture stop)."""
        if self._buffer and any(self._lexical_acc.rates.get(k, 0) for k in ("hedging", "certainty")):
            snap = self._snapshot()
            self._reset_for_next_window()
            return [snap]
        return []

    def reset(self) -> None:
        self._buffer.clear()
        self._count = 0
        self._lexical_acc = LexicalFeatures.neutral()
        self._window_start = None
        self._last_speaker_id = None
        self._speaker_names.clear()

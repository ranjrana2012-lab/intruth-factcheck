"""Lexical feature extraction — ported from InTruth's lexical-features.js / extractLexical.

Computes hedging/certainty/emotional/etc. rates over text. Used as a signal appended to
claim-evaluation prompts (helps the LLM weight rhetoric vs. factual commitment).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Word lists — verbatim from InTruth service-worker-ex.js
HEDGING_WORDS = [
    "think", "believe", "maybe", "perhaps", "probably", "might", "could",
    "seem", "appears", "guess", "suppose", "somewhat",
]
CERTAINTY_WORDS = [
    "definitely", "certainly", "absolutely", "always", "never", "clearly",
    "obviously", "undoubtedly", "exactly", "proven",
]
FILLER_WORDS = ["um", "uh", "like", "basically", "actually", "literally", "right", "okay"]
EMOTIONAL_WORDS = [
    "disaster", "terrible", "horrible", "amazing", "incredible", "great",
    "awful", "fantastic", "disgusting", "wonderful", "worst", "best",
]
EXCLUSIVE_WORDS = ["but", "except", "however", "although", "unless", "without", "exclude"]
FP_SINGULAR = ["i", "me", "my", "mine", "myself"]


@dataclass
class LexicalFeatures:
    rates: dict[str, int] = field(default_factory=dict)
    words_per_second: float | None = None
    word_count: int = 0

    @classmethod
    def neutral(cls) -> "LexicalFeatures":
        return cls(
            rates={
                "hedging": 0,
                "certainty": 0,
                "filler": 0,
                "emotional": 0,
                "exclusive": 0,
                "firstPersonSg": 0,
            },
            words_per_second=None,
            word_count=0,
        )


def extract_lexical(text: str) -> LexicalFeatures:
    """Port of InTruth's extractLexical. Returns per-category rates as % of word count."""
    words = [w for w in text.lower().split() if w]
    total = len(words) or 1

    def rate(word_list: list[str]) -> int:
        return round(sum(1 for w in words if any(h in w for h in word_list)) / total * 100)

    return LexicalFeatures(
        rates={
            "hedging": rate(HEDGING_WORDS),
            "certainty": rate(CERTAINTY_WORDS),
            "filler": rate(FILLER_WORDS),
            "emotional": rate(EMOTIONAL_WORDS),
            "exclusive": rate(EXCLUSIVE_WORDS),
            "firstPersonSg": round(sum(1 for w in words if w in FP_SINGULAR) / total * 100),
        },
        words_per_second=None,
        word_count=total,
    )


def build_lexical_summary(f: LexicalFeatures) -> str:
    """Port of InTruth's buildLexicalSummary — a one-line description for the prompt."""
    r = f.rates
    notes = []
    if r.get("hedging", 0) > 5:
        notes.append(f"hedging language ({r['hedging']}%)")
    if r.get("certainty", 0) > 5:
        notes.append(f"certainty markers ({r['certainty']}%)")
    if r.get("filler", 0) > 5:
        notes.append(f"filler words ({r['filler']}%)")
    if r.get("emotional", 0) > 5:
        notes.append(f"emotional language ({r['emotional']}%)")
    if r.get("exclusive", 0) > 5:
        notes.append(f"qualifying words ({r['exclusive']}%)")
    if r.get("firstPersonSg", 0) > 5:
        notes.append(f"first-person singular ({r['firstPersonSg']}%)")
    if f.words_per_second:
        pace = "fast" if f.words_per_second > 3.5 else "slow" if f.words_per_second < 2 else "moderate"
        notes.append(f"speech rate {f.words_per_second} w/s ({pace})")
    return f"Features detected: {', '.join(notes)}." if notes else "Neutral delivery."

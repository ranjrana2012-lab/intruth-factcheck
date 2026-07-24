"""Claim deduplication — ported from InTruth's isDuplicate / normalizeClaimKey.

Prevents re-checking the same claim restated. Three mechanisms (all from InTruth):
  1. Exact normalized-key match (lowercase, strip non-alnum, keep words ≥4 chars, sort)
  2. Keyword overlap ≥ 0.35 (Jaccard-ish over the normalized words)
  3. Monetary-figure guard: if two claims share the same $-figure, treat as duplicate
     (prevents re-checking "$2.4 billion" stated in different words)
"""
from __future__ import annotations

import re
import time

# Monetary regex — verbatim from InTruth
_MONEY_RE = re.compile(r"\$[\d,.]+(?:\s*(?:trillion|billion|million|thousand))?", re.IGNORECASE)


class ClaimDeduplicator:
    """TTL-based dedup. Thread-unsafe by design — the engine drives it from one async task."""

    def __init__(self, ttl_ms: int = 200_000, overlap_threshold: float = 0.35):
        self.ttl_ms = ttl_ms
        self.overlap_threshold = overlap_threshold
        # key → (timestamp, original_claim)
        self._recent: dict[str, tuple[float, str]] = {}

    @staticmethod
    def normalize_key(claim: str) -> str:
        """Lowercase → strip non-alphanumeric → keep words ≥4 chars → sort → join."""
        cleaned = re.sub(r"[^a-z0-9\s]", "", claim.lower())
        words = [w for w in cleaned.split() if len(w) >= 4]
        return " ".join(sorted(words))

    @staticmethod
    def _extract_figures(claim: str) -> set[str]:
        return {m.replace(",", "").replace(" ", "").lower() for m in _MONEY_RE.findall(claim)}

    def _prune_expired(self, now: float) -> None:
        expired = [k for k, (t, _) in self._recent.items() if now - t > self.ttl_ms]
        for k in expired:
            del self._recent[k]

    def is_duplicate(self, claim: str) -> bool:
        """Return True if `claim` is a restatement of a recently-seen claim."""
        key = self.normalize_key(claim)
        now = time.time() * 1000  # match InTruth's ms-based timestamps
        self._prune_expired(now)

        # 1. Exact normalized match
        if key in self._recent:
            return True

        key_words = set(key.split())
        figures = self._extract_figures(claim)

        for k, (_t, orig_claim) in self._recent.items():
            k_words = k.split()
            # 2. Keyword overlap
            if key_words:
                overlap = sum(1 for w in k_words if w in key_words)
                if overlap / max(len(key_words), len(k_words)) >= self.overlap_threshold:
                    return True
            # 3. Shared monetary figures
            if figures:
                orig_figures = self._extract_figures(orig_claim)
                if figures & orig_figures:
                    return True

        self._recent[key] = (now, claim)
        return False

    def recent_claims(self, limit: int = 15) -> list[str]:
        """Return the most recent original claims (for the 'already checked' prompt context)."""
        items = sorted(self._recent.values(), key=lambda v: v[0])
        return [c for _, c in items[-limit:]]

    def clear(self) -> None:
        self._recent.clear()

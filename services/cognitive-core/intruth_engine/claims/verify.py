"""Verify orchestrator — the fact-check pipeline tying everything together.

Flow (ported from InTruth's evaluateClaims + groundAndUpdate, two-pass):
  1. Claim extraction (fast LLM pass) → list of {claim, verdict, speaker}
  2. Dedup filter (drop already-checked claims)
  3. Emit pending_claim events for each (low latency — show the user something fast)
  4. Parallel per-claim web retrieval (Tavily/SearXNG) + grounded re-judge (LLM)
  5. Apply verdict taxonomy + inversion detection + source bias
  6. Emit final verdict events

PII redaction runs on every transcript window BEFORE the LLM sees it, and on every claim
before it's published — defense in depth.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from ..bus import bus
from ..config import verify_config
from ..llm import chat_completion, extract_json_array, extract_json_object
from ..pii import redact
from ..protocol import ClaimEvent, DropVerdictEvent, Evidence, VerdictEvent
from .dedup import ClaimDeduplicator
from .prompts import (
    EVALUATE_SYSTEM,
    EVALUATE_USER_TEMPLATE,
    GROUNDED_SYSTEM,
    GROUNDED_USER_TEMPLATE,
    build_evidence_block,
)
from .window import WindowSnapshot

log = logging.getLogger(__name__)

# Make the mcp-servers factcheck package importable (it's a sibling, not a submodule)
_MCP_PATH = Path(__file__).resolve().parents[4] / "mcp-servers" / "mcp-factcheck"
if str(_MCP_PATH) not in sys.path:
    sys.path.insert(0, str(_MCP_PATH))

from retrieval import filter_results, get_provider  # noqa: E402
from bias import annotate_sources  # noqa: E402


@dataclass
class ExtractedClaim:
    claim: str
    verdict: str
    speaker: str | None = None


class VerifyPipeline:
    """Stateful per-session pipeline. One instance per capture session."""

    def __init__(self, source: str = "desktop_audio", language: str = "en"):
        self.source = source
        self.language = language
        self.dedup = ClaimDeduplicator(
            ttl_ms=verify_config().claim_dedup_ms,
            overlap_threshold=verify_config().claim_overlap_threshold,
        )
        self._provider = get_provider()
        log.info("verify pipeline ready (retrieval=%s, lang=%s)", self._provider.name, language)

    async def process_window(self, snap: WindowSnapshot, event_date: str | None = None) -> None:
        """Run the full two-pass verify on a filled sentence window."""
        # ── PII redaction FIRST (the LLM never sees un-redacted text) ─────────
        safe_transcript = redact(snap.context_text)
        safe_lexical = redact(snap.lexical_summary)

        already = "\n".join(f"- {c}" for c in self.dedup.recent_claims()) or "(none yet)"
        user_msg = EVALUATE_USER_TEMPLATE.format(
            context_header=self._context_header(snap),
            transcript=safe_transcript,
            already_checked=already,
            lexical_context=f"Lexical analysis: {safe_lexical}" if safe_lexical else "",
        )

        # ── Pass 1: claim extraction ──────────────────────────────────────────
        try:
            resp = await chat_completion(EVALUATE_SYSTEM, user_msg, temperature=0.0, max_tokens=2048)
        except Exception:
            log.exception("claim extraction LLM call failed")
            return

        raw_claims = extract_json_array(resp.text)
        # filter: must have claim + verdict; drop UNVERIFIABLE at extract (InTruth behavior);
        # drop duplicates
        candidates: list[ExtractedClaim] = []
        for c in raw_claims:
            claim_text = (c.get("claim") or "").strip()
            verdict = (c.get("verdict") or "").strip().upper()
            if not claim_text or not verdict or verdict == "UNVERIFIABLE":
                continue
            safe_claim = redact(claim_text)  # defense in depth
            if self.dedup.is_duplicate(safe_claim):
                continue
            candidates.append(ExtractedClaim(claim=safe_claim, verdict=verdict, speaker=c.get("speaker")))

        if not candidates:
            return

        # ── Emit pending claims immediately (low-latency UX) ──────────────────
        for c in candidates:
            await bus.publish(
                ClaimEvent(claim=c.claim, speaker=c.speaker, source=self.source)  # type: ignore[arg-type]
            )

        # ── Pass 2: parallel grounding per claim ──────────────────────────────
        await asyncio.gather(
            *(self._ground_and_judge(c, snap, safe_transcript, event_date) for c in candidates),
            return_exceptions=True,
        )

    async def _ground_and_judge(
        self,
        claim: ExtractedClaim,
        snap: WindowSnapshot,
        safe_transcript: str,
        event_date: str | None,
    ) -> None:
        """Retrieve evidence for one claim, re-judge, emit final verdict."""
        try:
            results = await self._provider.search(claim.claim, n=8)
        except Exception:
            log.exception("retrieval failed for claim: %s", claim.claim[:60])
            results = type(results)()  # empty

        organic = filter_results(
            results.organic,
            event_date=event_date,
            max_results=verify_config().max_sources_per_claim,
        )

        # No evidence → finalize as the preliminary verdict (don't leave card pending)
        if not organic and not results.answer_box and not results.knowledge_graph:
            await bus.publish(
                VerdictEvent(  # type: ignore[arg-type]
                    claim=claim.claim,
                    verdict=claim.verdict,
                    confidence="LOW",
                    explanation="No web evidence found.",
                    speaker=claim.speaker,
                    evidence=[],
                    sources=[],
                )
            )
            return

        evidence_block = build_evidence_block(
            answer_box=vars(results.answer_box) if results.answer_box else None,
            knowledge_graph=vars(results.knowledge_graph) if results.knowledge_graph else None,
            organic=[vars(o) for o in organic],
        )
        grounded_user = GROUNDED_USER_TEMPLATE.format(
            context_header=self._context_header(snap),
            transcript=safe_transcript,
            claim=claim.claim,
            preliminary_verdict=claim.verdict,
            evidence_block=evidence_block,
            lexical_context=f"Lexical analysis: {redact(snap.lexical_summary)}" if snap.lexical_summary else "",
        )

        try:
            resp = await chat_completion(GROUNDED_SYSTEM, grounded_user, temperature=0.0, max_tokens=2048)
        except Exception:
            log.exception("grounded LLM call failed for claim: %s", claim.claim[:60])
            await bus.publish(
                VerdictEvent(claim=claim.claim, verdict=claim.verdict, confidence="LOW", speaker=claim.speaker)  # type: ignore[arg-type]
            )
            return

        parsed = extract_json_object(resp.text)
        if not parsed or not parsed.get("verdict"):
            return
        verdict = parsed["verdict"].strip().upper()
        # InTruth: drop UNVERIFIABLE from grounded pass
        if verdict == "UNVERIFIABLE":
            await bus.publish(DropVerdictEvent(claim=claim.claim))  # type: ignore[arg-type]
            return

        # InTruth: inversion detection — drop misattributed claims
        explanation = parsed.get("explanation", "")
        expl_lower = explanation.lower()
        if any(
            trigger in expl_lower
            for trigger in ("transcript shows", "inverted", "not herself", "not himself")
        ):
            await bus.publish(DropVerdictEvent(claim=claim.claim))  # type: ignore[arg-type]
            return

        sources = [o.url for o in organic if o.url]
        await bus.publish(
            VerdictEvent(  # type: ignore[arg-type]
                claim=claim.claim,
                verdict=verdict,
                confidence=(parsed.get("confidence") or "HIGH").upper(),
                explanation=explanation,
                speaker=parsed.get("speaker") or claim.speaker,
                evidence=[
                    Evidence(title=o.title, url=o.url, snippet=o.snippet, date=o.date, kind="organic")
                    for o in organic
                ],
                sources=sources,
                bias=annotate_sources(sources),
            )
        )

    def _context_header(self, snap: WindowSnapshot) -> str:
        parts = []
        if snap.dominant_speaker:
            parts.append(f"Current speaker: {snap.dominant_speaker}.")
        if snap.opponent_name:
            parts.append(f"Opponent: {snap.opponent_name}.")
        if self.language != "en":
            parts.append(
                f"LANGUAGE: transcript is in {self.language}. Write claim+explanation in {self.language}; "
                "only verdict labels stay in English."
            )
        return (" ".join(parts) + "\n\n") if parts else ""

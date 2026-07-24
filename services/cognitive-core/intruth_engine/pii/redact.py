"""Presidio PII redaction — deterministic gate before any storage or cloud egress.

Why this exists: ambient capture WILL see passwords, credit cards, SSNs, names, addresses
in transcripts. We must strip them BEFORE the text goes to the LLM (Ollama Cloud) or into
the SQLite store. This is the privacy spine of the whole system.

Presidio uses NER + regex recognizers. On first use it downloads default model weights.
We keep the redactor lazy-loaded so the engine starts fast.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

# Regex pre-filter for high-sensitivity patterns Presidio might miss or to harden the gate.
# These run unconditionally (no model needed) as a first defensive layer.
_HARD_REGEXES = {
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "SSN_US": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "API_KEY_LIKE": re.compile(r"\b(?:sk-|oll_|gho_|ghp_|AKIA|tvly-)[A-Za-z0-9_\-]{10,}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
}


@lru_cache(maxsize=1)
def _get_anonymizer():
    """Lazy-load Presidio analyzer + anonymizer. Returns None if unavailable."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        log.info("presidio PII redactor loaded")
        return analyzer, anonymizer
    except Exception as e:
        log.warning("presidio unavailable (pip install presidio-analyzer presidio-anonymizer): %s", e)
        return None


def redact(text: str) -> str:
    """Redact PII from `text`. Always runs the regex layer; uses Presidio NER if available.

    Returns text with sensitive entities replaced by tokens like <CREDIT_CARD>, <PERSON>, etc.
    Safe to call on empty/None.
    """
    if not text:
        return text

    # Layer 1: hard regex (always on, no deps)
    out = text
    for label, pattern in _HARD_REGEXES.items():
        out = pattern.sub(f"<{label}>", out)

    # Layer 2: Presidio NER (if installed)
    anon = _get_anonymizer()
    if anon is not None:
        analyzer, anonymizer = anon
        try:
            results = analyzer.analyze(
                text=out,
                entities=["PERSON", "PHONE_NUMBER", "US_SSN", "EMAIL_ADDRESS", "URL"],
                language="en",
            )
            if results:
                from presidio_anonymizer.entities import OperatorConfig

                out = anonymizer.anonymize(
                    text=out,
                    analyzer_results=results,
                    operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})},
                ).text
        except Exception:
            log.debug("presidio analysis failed on a chunk; regex layer already applied", exc_info=True)

    return out

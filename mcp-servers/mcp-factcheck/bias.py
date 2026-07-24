"""Source-bias annotation — vendored MBFC-style domain → bias/credibility lookup.

InTruth relied on an external backend (intruth-backend.vercel.app/api/bias). We vendor a
small built-in mapping so the fact-check tool is self-contained with no external
dependency. The full MBFC dataset (https://github.com/idiap/Factual-Reporting-and-...)
can be dropped in as data/mbfc.csv for broader coverage.

Bias labels: left | left-center | center | right-center | right
Credibility: high | mostly-factual | mixed | low
"""
from __future__ import annotations

# Robust import: works whether this is imported as a package member or via sys.path
try:
    from .retrieval.filter import domain_of
except ImportError:
    from retrieval.filter import domain_of

# Seed mapping of high-traffic domains. Extend via data/mbfc.csv when present.
_SEED_BIAS: dict[str, dict[str, str]] = {
    "reuters.com": {"bias": "center", "credibility": "high"},
    "apnews.com": {"bias": "center", "credibility": "high"},
    "bbc.com": {"bias": "center", "credibility": "high"},
    "bbc.co.uk": {"bias": "center", "credibility": "high"},
    "nytimes.com": {"bias": "left-center", "credibility": "high"},
    "washingtonpost.com": {"bias": "left-center", "credibility": "high"},
    "cnn.com": {"bias": "left-center", "credibility": "mostly-factual"},
    "npr.org": {"bias": "center", "credibility": "high"},
    "wsj.com": {"bias": "right-center", "credibility": "high"},
    "foxnews.com": {"bias": "right", "credibility": "mixed"},
    "bloomberg.com": {"bias": "center", "credibility": "high"},
    "economist.com": {"bias": "center", "credibility": "high"},
    "theguardian.com": {"bias": "left-center", "credibility": "high"},
    "politifact.com": {"bias": "center", "credibility": "high"},
    "factcheck.org": {"bias": "center", "credibility": "high"},
    "snopes.com": {"bias": "center", "credibility": "high"},
    "ap.org": {"bias": "center", "credibility": "high"},
    "nature.com": {"bias": "center", "credibility": "high"},
    "science.org": {"bias": "center", "credibility": "high"},
    "govtrack.us": {"bias": "center", "credibility": "high"},
    "congress.gov": {"bias": "center", "credibility": "high"},
    "bls.gov": {"bias": "center", "credibility": "high"},  # US Bureau of Labor Statistics
    "bea.gov": {"bias": "center", "credibility": "high"},  # Bureau of Economic Analysis
    "federalreserve.gov": {"bias": "center", "credibility": "high"},
    "cdc.gov": {"bias": "center", "credibility": "high"},
    "who.int": {"bias": "center", "credibility": "high"},
    "wikipedia.org": {"bias": "center", "credibility": "mostly-factual"},
}

# Lazily-loaded full CSV if present
_csv_loaded = False
_csv_map: dict[str, dict[str, str]] = {}


def _ensure_csv_loaded() -> None:
    global _csv_loaded
    if _csv_loaded:
        return
    _csv_loaded = True
    from pathlib import Path

    csv_path = Path(__file__).parent / "data" / "mbfc.csv"
    if not csv_path.exists():
        return
    import csv

    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = (row.get("url") or row.get("domain") or "").lower()
                if not url:
                    continue
                d = url.split("/")[-1] or url
                d = d.replace("www.", "")
                _csv_map[d] = {
                    "bias": (row.get("bias") or "").lower(),
                    "credibility": (row.get("factual") or row.get("credibility") or "").lower(),
                }
        # CSV entries override seed
    except Exception:
        pass


def lookup_bias(url: str) -> dict[str, str] | None:
    """Return {bias, credibility} for a source URL, or None if unknown."""
    _ensure_csv_loaded()
    d = domain_of(url).replace("www.", "")
    if not d:
        return None
    return _csv_map.get(d) or _SEED_BIAS.get(d)


def annotate_sources(urls: list[str]) -> list[dict]:
    """Annotate a list of source URLs with bias pills."""
    out = []
    for url in urls:
        b = lookup_bias(url)
        if b:
            out.append({"domain": domain_of(url), "url": url, **b})
    return out

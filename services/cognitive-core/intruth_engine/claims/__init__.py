"""Claims package: windowing, dedup, lexical features, prompts, extraction."""
from .dedup import ClaimDeduplicator
from .lexical import LexicalFeatures, build_lexical_summary, extract_lexical
from .window import SentenceWindow, WindowSnapshot

__all__ = [
    "ClaimDeduplicator",
    "SentenceWindow",
    "WindowSnapshot",
    "LexicalFeatures",
    "extract_lexical",
    "build_lexical_summary",
]

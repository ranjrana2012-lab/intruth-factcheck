"""Retrieval package — swappable web-search providers + source filtering."""
from .base import AnswerBox, KnowledgeGraph, OrganicResult, RetrievalProvider, SearchResults
from .filter import BLOCKED_DOMAINS, domain_of, filter_results, is_blocked
from .searxng import SearXNGProvider
from .tavily import TavilyProvider


def get_provider(name: str | None = None) -> RetrievalProvider:
    """Build the configured provider. Falls back through Tavily → SearXNG → None."""
    from intruth_engine.config import get_settings, verify_config

    cfg = verify_config()
    name = name or cfg.retrieval_provider
    settings = get_settings()

    if name == "tavily" and settings.tavily_api_key:
        return TavilyProvider()
    if name == "searxng" or (name == "tavily" and not settings.tavily_api_key):
        return SearXNGProvider()
    if name == "both":
        # Return Tavily primary; the orchestrator can also call SearXNG as backup
        return TavilyProvider() if settings.tavily_api_key else SearXNGProvider()
    # default: whatever has credentials
    if settings.tavily_api_key:
        return TavilyProvider()
    return SearXNGProvider()


__all__ = [
    "RetrievalProvider",
    "SearchResults",
    "OrganicResult",
    "AnswerBox",
    "KnowledgeGraph",
    "TavilyProvider",
    "SearXNGProvider",
    "get_provider",
    "filter_results",
    "is_blocked",
    "domain_of",
    "BLOCKED_DOMAINS",
]

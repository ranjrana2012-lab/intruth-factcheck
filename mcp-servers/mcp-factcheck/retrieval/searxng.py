"""SearXNG retrieval provider — privacy-respecting self-hosted metasearch.

Run locally via Docker: `docker run -p 8080:8080 searxng/searxng`. Aggregates 70+ search
engines with tracking stripped. No API key, fully private — ideal for the fact-check use
case where we don't want to leak query profiles.

SearXNG returns organic results only (no Answer Box / Knowledge Graph), so those fields
stay None. The verify pipeline handles that gracefully.
"""
from __future__ import annotations

import logging

import httpx

from intruth_engine.config import get_settings

from .base import OrganicResult, RetrievalProvider, SearchResults

log = logging.getLogger(__name__)


class SearXNGProvider(RetrievalProvider):
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or get_settings().searxng_url or "http://localhost:8080").rstrip("/")

    @property
    def name(self) -> str:
        return "searxng"

    async def search(self, query: str, n: int = 8) -> SearchResults:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self.base_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "safesearch": 0,
                        "categories": "general",
                    },
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            log.warning("searxng search failed (is it running? %s): %s", self.base_url, query[:80])
            return SearchResults()

        organic = [
            OrganicResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
            )
            for r in data.get("results", [])[:n]
        ]
        return SearchResults(organic=organic)

"""Tavily retrieval provider — AI-oriented web search with an 'answer' field.

Quick-start option (free tier available at https://tavily.com). Returns an answer
synthesis + organic results. We map these onto our SearchResults shape.

No key yet? The pipeline falls back to SearXNG (see searxng.py) or returns empty results
gracefully — claim extraction still works, verdicts just say UNVERIFIABLE.
"""
from __future__ import annotations

import logging

import httpx

from intruth_engine.config import get_settings

from .base import AnswerBox, KnowledgeGraph, OrganicResult, RetrievalProvider, SearchResults

log = logging.getLogger(__name__)


class TavilyProvider(RetrievalProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_settings().tavily_api_key

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, query: str, n: int = 8) -> SearchResults:
        if not self.api_key:
            log.debug("tavily: no API key — returning empty results")
            return SearchResults()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": n,
                        "include_answer": True,
                        "include_raw_content": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            log.exception("tavily search failed for query: %s", query[:80])
            return SearchResults()

        organic = [
            OrganicResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
            )
            for r in data.get("results", [])[:n]
        ]
        answer_box = None
        if data.get("answer"):
            answer_box = AnswerBox(answer=data["answer"], title="Tavily Answer")
        return SearchResults(organic=organic, answer_box=answer_box)

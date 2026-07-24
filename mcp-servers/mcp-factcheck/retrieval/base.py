"""RetrievalProvider interface — swappable web-search backends for evidence gathering.

Every provider returns SearchResults in the same shape so the verify pipeline is
provider-agnostic. Tavily = quick cloud start (API key); SearXNG = private/self-hosted
(Docker, no key). Switch via config.win.yaml `verify.retrieval_provider`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OrganicResult:
    url: str
    title: str = ""
    snippet: str = ""
    date: str = ""


@dataclass
class AnswerBox:
    answer: str
    title: str = ""
    url: str = ""


@dataclass
class KnowledgeGraph:
    description: str
    title: str = ""


@dataclass
class SearchResults:
    organic: list[OrganicResult] = field(default_factory=list)
    answer_box: AnswerBox | None = None
    knowledge_graph: KnowledgeGraph | None = None


class RetrievalProvider(ABC):
    """Abstract web-search provider."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def search(self, query: str, n: int = 8) -> SearchResults:
        """Search the web for `query`. Returns ranked results."""
        ...

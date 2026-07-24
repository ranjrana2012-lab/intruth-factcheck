"""LLM client — talks to Ollama Cloud (OpenAI-compatible) for claim extraction + verdicts.

Ollama Cloud exposes an OpenAI-compatible /v1/chat/completions endpoint, so this client
also works unchanged against local Ollama, OpenAI, or any compatible gateway — just change
the base URL + key in .env. The model name is config-driven.

Provider switching is one .env line: OLLAMA_BASE_URL + OLLAMA_API_KEY. We don't import the
heavy SDKs; we call HTTP directly with httpx for a thin, fast dependency footprint.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from .config import get_settings, llm_config

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    raw: dict


async def chat_completion(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 60.0,
) -> LLMResponse:
    """Call the configured LLM (Ollama Cloud by default). OpenAI-compatible request."""
    settings = get_settings()
    llm_cfg, _ = llm_config()
    base_url = (settings.ollama_base_url or "https://api.ollama.com").rstrip("/")
    model = model or llm_cfg.model
    api_key = settings.ollama_api_key or "ollama"  # local Ollama needs no key

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # OpenAI-compatible shape
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error("LLM: unexpected response shape: %s", json.dumps(data)[:300])
        raise RuntimeError(f"LLM response missing content: {e}") from e
    return LLMResponse(text=text, raw=data)


def extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from model output (tolerant of code fences / preamble).

    Port of InTruth's parseArray: find the first '[' and last ']'.
    """
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from model output (for the grounded single-verdict pass)."""
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

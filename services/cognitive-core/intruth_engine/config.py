"""Configuration: environment variables (`.env`) layered with optional YAML.

Keys/secrets come from the environment (`.env`, never committed). Behavioural settings
(window size, autonomy levels, source allowlists) come from `config.win.yaml`. The env
file takes precedence for secrets; YAML drives the policy.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = 2 levels up from this file (services/cognitive-core/intruth_engine/)
REPO_ROOT = Path(__file__).resolve().parents[2]


class LLMConfig(BaseModel):
    provider: str = "ollama_cloud"  # ollama_cloud | ollama_local | anthropic | openai
    model: str = "qwen2.5:7b"
    extraction_model: str | None = None
    verdict_model: str | None = None


class ASRConfig(BaseModel):
    provider: str = "faster_whisper"
    model: str = "small"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"  # auto | int8 | float16


class AudioCaptureConfig(BaseModel):
    enabled: bool = True
    adapter: str = "wasapi_loopback"  # wasapi_loopback | screenpipe | pyaudio
    sample_rate_hz: int = 16000
    microphone_enabled: bool = True
    vad_sensitivity: float = 0.5


class VerifyConfig(BaseModel):
    retrieval_provider: str = "tavily"  # tavily | searxng | both
    max_sources_per_claim: int = 4
    block_partisan_sources: bool = True
    window_size: int = 4
    window_keep: int = 15
    claim_dedup_ms: int = 200_000
    claim_overlap_threshold: float = 0.35


class EngineSettings(BaseSettings):
    """Secrets + runtime params from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    ollama_base_url: str = ""
    ollama_api_key: str = ""
    llm_model: str = "qwen2.5:7b"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Retrieval
    tavily_api_key: str = ""
    searxng_url: str = ""

    # STT (Deepgram only for the legacy extension path)
    deepgram_api_key: str = ""

    # Engine runtime
    engine_host: str = "127.0.0.1"
    engine_port: int = 8765
    asr_device: str = "auto"
    asr_model: str = "small"
    vad_threshold: float = 0.5


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> EngineSettings:
    return EngineSettings()


@lru_cache(maxsize=1)
def get_yaml_config() -> dict:
    """Load config.win.yaml (falls back to the .example template if the real one is absent)."""
    real = REPO_ROOT / "config.win.yaml"
    example = REPO_ROOT / "config.win.example.yaml"
    return _load_yaml(real) or _load_yaml(example)


def llm_config() -> tuple[LLMConfig, EngineSettings]:
    """Resolve the active LLM config by merging YAML `models.llm` over env defaults."""
    settings = get_settings()
    yaml_cfg = get_yaml_config().get("models", {}).get("llm", {})
    merged = {
        "provider": yaml_cfg.get("provider", "ollama_cloud"),
        "model": yaml_cfg.get("model", settings.llm_model),
        "extraction_model": yaml_cfg.get("extraction_model"),
        "verdict_model": yaml_cfg.get("verdict_model"),
    }
    return LLMConfig(**merged), settings


def asr_config() -> ASRConfig:
    yaml_cfg = get_yaml_config().get("models", {}).get("stt_engine", {})
    settings = get_settings()
    return ASRConfig(
        provider=yaml_cfg.get("provider", "faster_whisper"),
        model=yaml_cfg.get("model", settings.asr_model),
        device=yaml_cfg.get("device", settings.asr_device),
        compute_type=yaml_cfg.get("compute_type", "auto"),
    )


def audio_config() -> AudioCaptureConfig:
    yaml_cfg = get_yaml_config().get("perception", {}).get("audio_capture", {})
    settings = get_settings()
    return AudioCaptureConfig(
        enabled=yaml_cfg.get("enabled", True),
        adapter=yaml_cfg.get("adapter", "wasapi_loopback"),
        sample_rate_hz=yaml_cfg.get("sample_rate_hz", 16000),
        microphone_enabled=yaml_cfg.get("microphone_enabled", True),
        vad_sensitivity=yaml_cfg.get("vad_sensitivity", settings.vad_threshold),
    )


def verify_config() -> VerifyConfig:
    yaml_cfg = get_yaml_config().get("verify", {})
    return VerifyConfig(**yaml_cfg) if yaml_cfg else VerifyConfig()


def data_dir() -> Path:
    """Directory for the SQLite DB (claims + verdicts only). Created on demand."""
    d = REPO_ROOT / "services" / "cognitive-core" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_device(requested: str) -> str:
    """Resolve 'auto' → 'cuda' if a CUDA GPU is available, else 'cpu'.

    Checks CTranslate2 (faster-whisper's backend) rather than torch, since torch is only
    present for Silero VAD and may ship without CUDA libs. CTranslate2 has its own CUDA
    runtime and is what actually accelerates Whisper on the GPU.
    """
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"

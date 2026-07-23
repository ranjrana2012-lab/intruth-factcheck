"""InTruth FactCheck cognitive core.

Local-first pipeline: capture (WASAPI/mic/screen) → Silero VAD → faster-whisper →
PII redaction → claim windowing/dedup → verdict synthesis → SQLite + WebSocket fan-out.

See docs/intruth-logic-extraction.md for the ported InTruth pipeline behavior.
"""

__version__ = "0.1.0"

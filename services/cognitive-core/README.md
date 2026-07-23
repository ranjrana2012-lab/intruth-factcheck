# InTruth Cognitive Core

The Python engine: capture → VAD → ASR → (M2: PII redaction → claim extraction → verdict
synthesis) → SQLite + WebSocket fan-out.

## Run

```bash
uv venv
uv pip install -e ".[pii,dev]"
uv run python -m intruth_engine.server
# → http://127.0.0.1:8765
```

Configure via the repo-root `.env` and `config.win.yaml` (see `.env.example` and
`config.win.example.yaml`).

See `docs/intruth-logic-extraction.md` for the ported InTruth pipeline behavior.

# InTruth FactCheck — Local-First Ambient Fact-Checking Assistant

A fork of [`rpanigrahi222/intruth-factcheck`](https://github.com/rpanigrahi222/intruth-factcheck),
re-architected from a **Chrome tab-capture extension** into a **local-first native desktop
assistant** that can fact-check **anything your computer hears or sees** — not just one
browser tab.

> **Status:** 🚧 Active development. M0 (scaffold) complete. See [Roadmap](#roadmap).

---

## What it does

Real-time, evidence-backed fact-checking of ambient speech and on-screen content:

- 🎙️ **System audio** (WASAPI loopback) — podcasts, videos, calls, anything playing
- 🎤 **Microphone** — conversations, lectures, your own voice
- 🖥️ **Screen content** — OCR of text shown on screen
- 📱 **Phone** (companion app) — audio streamed from your phone to your desktop

Claims are extracted as they're spoken, verified against web evidence, and shown as
color-coded verdicts: **TRUE / SUBSTANTIALLY TRUE / FALSE / MISLEADING / UNVERIFIABLE**,
each with citations and source-bias tags.

---

## Architecture — local capture + cloud reasoning

The key design decision: **capture and transcription are local and private; only
Presidio-redacted claim text ever leaves your machine** (for an LLM verdict). Raw audio is
never persisted and never transmitted.

```
┌──────────────── ZONE 1 — Native capture (Rust/Screenpipe) ──────────────────┐
│  WASAPI loopback (system audio) + microphone + DXGI screen + OCR            │
│  App-exclusion auto-pause (1Password/banking) · raw buffers volatile         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ 16kHz mono Int16 PCM (WS)
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                    ZONE 2 — Cognitive core (Python/FastAPI)                  │
│  Silero VAD → faster-whisper (your GPU) → Presidio PII redaction            │
│  → claim windowing + dedup → verdict synthesis (Ollama Cloud)               │
│  → SQLite (claims + verdicts only) → WebSocket fan-out                      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ redacted claim text only
┌───────────────────────────────▼─────────────────────────────────────────────┐
│              ZONE 3 — Isolated MCP execution (per-action approval)          │
│  mcp-factcheck (first tool) · browser/filesystem/desktop (later)            │
│  Autonomy Levels 0–4 gate every action with external effects                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why this split

| Layer | Where | Why |
|---|---|---|
| Capture | **Local** (Screenpipe/WASAPI/DXGI) | The sensory organ — can't stream everything to the cloud |
| VAD + STT | **Local, your GPU**, event-gated | Only runs on detected speech → light load, audio never leaves the machine |
| PII redaction | **Local** (Presidio) | Deterministic gate before any storage or egress |
| LLM (claim/verdict) | **Ollama Cloud** (config-driven) | Bursty, 5–15s budget, no local 8B model hogging VRAM → desktop stays fast |
| Raw audio | Local, **volatile** | Discarded after transcription; never persisted |

---

## Repository layout

```
intruth-factcheck/
├── apps/desktop/              # Tauri v2 shell (Rust + WebView2): tray, hotkey, toast, supervisor
│   ├── src-tauri/
│   └── ui/                    # dashboard: live transcript + verdict feed
├── services/cognitive-core/   # Python: FastAPI orchestrator + ASR + claims + PII
│   └── intruth_engine/
├── mcp-servers/mcp-factcheck/ # InTruth's claim/verify logic as an MCP tool
│   ├── retrieval/             # RetrievalProvider: tavily.py, searxng.py
│   └── prompts.py             # fresh EVALUATE/GROUNDED prompts
├── adapters/                  # capture: screenpipe client, mic, screen OCR
├── policies/                  # autonomy levels, app-exclusions, retention
├── extension/realtime-factcheck/  # original InTruth MV3 extension (refactored as a client)
├── mobile/                    # Expo companion app
├── docs/intruth-logic-extraction.md  # faithful port notes for the InTruth pipeline
├── config.win.example.yaml
└── .env.example
```

---

## Quick start (once M1 lands)

```bash
# 1. Clone your fork
git clone https://github.com/ranjrana2012-lab/intruth-factcheck.git
cd intruth-factcheck

# 2. Configure secrets
cp .env.example .env             # fill in OLLAMA_API_KEY, TAVILY_API_KEY
cp config.win.example.yaml config.win.yaml

# 3. Run the cognitive core (M1)
cd services/cognitive-core
uv sync
uv run uvicorn intruth_engine.server:app --reload

# 4. Run the Tauri shell (M1b)
cd ../../apps/desktop
npm install
npm run tauri dev
```

Speak or play audio → see live transcripts and verdicts on the dashboard.

---

## Privacy & legal — read this before enabling always-on capture

**Always-on ambient capture is legally hazardous.** This project bakes in mitigations, but
**you are responsible for complying with your local laws.**

- **Two-party / all-party consent jurisdictions** (US: CA, FL, IL, MD, MA, MI, MT, NV, NH,
  PA, WA; plus EU/UK GDPR) — recording conversations without **all parties' consent** can be
  a **felony**. Ambient capture will inevitably record third parties.
- **Mitigations in this codebase:**
  - Transcription is **local**; raw audio is **discarded immediately after transcription** and **never persisted or transmitted**.
  - Only **Presidio-redacted claim text** leaves your machine (for the LLM verdict).
  - SQLite stores **claims + verdicts only** — no raw audio, no full third-party transcripts.
  - An **un-hideable tray indicator** shows capture state.
  - An **app-exclusion list** auto-pauses capture when 1Password/banking/private windows are foregrounded.
  - A **global pause hotkey** (`Win+Alt+J`) and explicit consent flow.
- **Mobile has no silent always-on mic.** iOS shows an unavoidable orange dot; Android
  requires a visible foreground-service notification. The phone companion is a foreground
  app you explicitly start — not invisible background recording.

If in doubt, get consent from everyone in earshot before enabling capture.

---

## Security model

The top threat for an always-on agent is **indirect prompt injection**: a malicious prompt
hidden in captured content (a webpage, a video's audio) that hijacks the agent into taking
harmful action. This architecture counters it structurally:

- **Three trust zones** (capture → cognitive core → isolated execution) separate
  observation from action.
- **MCP isolation** — every tool with external effects is a separate process; the LLM
  cannot execute arbitrary code.
- **Autonomy Levels 0–4** gate every action:
  - `0 Observe` — record/evaluate only, no egress
  - `1 Inform` — show a toast (fact-check verdicts live here)
  - `2 Prepare` — draft an action, halt for approval
  - `3 Execute` — run only after explicit user click
  - `4 Pre-authorized` — narrowly scoped reversible workflows
- **Network egress allowlist** blocks all outbound traffic except configured endpoints.

---

## Roadmap

- [x] **M0** — Fork, monorepo scaffold, extension `-ex` fix, docs, config templates
- [ ] **M1** — Cognitive core: FastAPI WS hub + WASAPI capture + Silero VAD + faster-whisper → live transcripts
- [ ] **M1b** — Tauri native shell: tray indicator, pause hotkey, toast, supervisor
- [ ] **M2** — Fact-check pipeline: Presidio + ported windowing/dedup + claim prompts + retrieval (Tavily/SearXNG) + Ollama Cloud synthesis + MCP factcheck tool
- [ ] **M3** — Three clients: desktop mic+screen OCR, browser extension refactor, Expo phone companion
- [ ] **M4** — Hardening: autonomy approval UI, app-exclusion, tests, installer packaging

**Deferred (architected-for, not MVP):** openWakeWord + Kokoro (voice interaction), vector
memory (LanceDB/mem0), OmniParser (GUI understanding), browser-use (automation). All plug
into the Zone-3 MCP boundary.

---

## Acknowledgements

Built on the excellent work of:
- **[rpanigrahi222/intruth-factcheck](https://github.com/rpanigrahi222/intruth-factcheck)** — the original real-time fact-checking extension whose claim/verdict pipeline we port
- **[mediar-ai/screenpipe](https://github.com/mediar-ai/screenpipe)**, **[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)**, **[snakers4/silero-vad](https://github.com/snakers4/silero-vad)**, **[ollama/ollama](https://github.com/ollama/ollama)**, **[data-privacy-stack/presidio](https://github.com/data-privacy-stack/presidio)**, **[tauri-apps/tauri](https://github.com/tauri-apps/tauri)**

## License

The original InTruth project is MIT-licensed (Copyright 2026 risha panigrahi). This fork
retains MIT licensing for the ported components.

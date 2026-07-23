# InTruth Logic Extraction Notes

Reference for porting the original `rpanigrahi222/intruth-factcheck` pipeline (a Chrome MV3
extension) into the native cognitive-core engine. The original is JS in the browser; the
engine is Python. These are the verbatim behaviors/heuristics to preserve.

**Source files (now in `extension/realtime-factcheck/src/`):**
- `background/service-worker.js` — the entire claim/verify pipeline (903 lines)
- `offscreen/offscreen.js` — audio capture → Deepgram WS (213 lines)
- `content/lexical-features.js` — commitment/hedging/certainty scoring
- `content/session-export.js` — HTML session report

---

## 1. Verdict taxonomy (preserve exactly)

Five labels, plus a LOW-certainty override that turns a card yellow:

```
TRUE | SUBSTANTIALLY TRUE | FALSE | MISLEADING | UNVERIFIABLE
```

Rules observed in `groundAndUpdate`:
- UNVERIFIABLE is **dropped from the grounded pass** — "either it's checkable or it isn't shown."
- The fast (Haiku) pass also filters out `verdict === 'UNVERIFIABLE'` before grounding.
- **No never-downgrade guard** (intentionally removed). The grounded verdict always wins,
  even if it contradicts the fast pass. Comment in source: this removed a "softening bias"
  — trusting grounded evidence raised PolitiFact/AP/FactCheck.org agreement +8pt and the
  share of problematic claims flagged +11pt, with FALSE precision ~93%.

## 2. Inversion detection (drop the card)

After the grounded pass, if `explanation.toLowerCase()` contains any of:
`'transcript shows'`, `'inverted'`, `'not herself'`, `'not himself'`, `'not harris'`,
`'not trump'` → the verdict is a hallucinated speaker attribution. Emit `DROP_VERDICT`.

## 3. Claim deduplication (`isDuplicate`)

- `CLAIM_DEDUP_MS = 200_000` (200s) TTL on a `recentClaims` map.
- `normalizeClaimKey(claim)`:
  lowercase → strip non-alphanumeric → keep words length ≥ 4 → sort → join with space.
- Exact-key match → duplicate.
- Keyword-overlap match: overlap / max(lenA, lenB) **>= 0.35** → duplicate.
- **Monetary figure guard** — regex `/\$[\d,.]+(?:\s*(?:trillion|billion|million|thousand))?/gi`,
  then `.replace(/[,\s]/g,'').toLowerCase()`. If a claim's figures intersect another claim's
  figures → duplicate. (Prevents re-checking the same "$2.4 billion" stated differently.)

## 4. Sentence windowing (`onNewSentence`)

- `WINDOW_SIZE = 4`, `WINDOW_KEEP = 15` (rolling buffer).
- Evaluation fires when `sentenceCount % WINDOW_SIZE === 0` **OR** on a speaker change
  mid-window (with ≥2 sentences already buffered).
- On fire, build a "dominant speaker" = the speakerId appearing most in the **current**
  window (last `WINDOW_SIZE` sentences), not the whole buffer.
- Lexical features are accumulated as a running sum and averaged over the sentence count
  at snapshot time (not re-computed over the window).

## 5. Lexical features (`extractLexical`)

Word lists (lowercase substring match):
- HEDGING: think, believe, maybe, perhaps, probably, might, could, seem, appears, guess, suppose, somewhat
- CERTAINTY: definitely, certainly, absolutely, always, never, clearly, obviously, undoubtedly, exactly, proven
- FILLER: um, uh, like, basically, actually, literally, right, okay
- EMOTIONAL: disaster, terrible, horrible, amazing, incredible, great, awful, fantastic, disgusting, wonderful, worst, best
- EXCLUSIVE: but, except, however, although, unless, without, exclude
- FIRST_PERSON_SG: i, me, my, mine, myself

Each rate = `round(count/total_words * 100)`. Summary builder flags any rate > 5%.
Speech rate: wordsPerSecond > 3.5 = "fast", < 2 = "slow", else "moderate".

## 6. Pronoun resolution (`resolveDismissivePronouns`)

Runs **before** the LLM sees the text. Rewrites dismissive framing so the speaker isn't
misattributed. Patterns (with `opponentName`):
- "coming from someone who" → "coming from {opp} who"
- "said by someone who" → "said by {opp} who"
- "from (a)? (man|woman|guy|person) who" → "from {opp} who"
- "from him/her/them" → "from {opp}"
- "that's rich coming from" → "that's rich coming from {opp},"
- "you('ve|have)? been found liable" → "{opp} has been found liable"
- legal tense rewrite for prosecuted|convicted|indicted|sentenced|charged|arrested|impeached

`getOpponentName(speakerName)` resolves via confirmed speaker map, else via
`parseSpeakersFromTitle`.

## 7. Speaker parsing from title (`parseSpeakersFromTitle`)

Three regex strategies, in order:
1. `"N role vs N role"` → `[(roleA cap), (roleB cap)]`
2. `"Name vs N Description"` → `[lastName, lastNonNoiseWord]`
3. split on `vs/and/&` → last capitalized non-noise word from each side

`SPEAKER_PARSE_NOISE`: debate, presidential, vp, vice, years (2016-2024), surrounded,
tonight, live, full, official.

## 8. Evidence block formatting (preserve order)

In `groundAndUpdate`, evidence is formatted for the LLM in this strict order:
1. `[Direct Answer]` — from answerBox (highest quality signal)
2. `[Knowledge Panel]` — from knowledgeGraph
3. `[1]` `[2]` … — organic results with title, date, url, snippet

## 9. Source filtering (`BLOCKED_DOMAINS` + date)

- Large blocklist: social media, partisan left/right, state media, conspiracy/low-cred,
  advocacy orgs, PDF repos. (Port verbatim — see source lines 61–106.)
- Date filter: drop sources dated **more than 1 year after** the event date
  (prevents future-knowledge contamination of historical claims).
- Cap organic results to top 4.

## 10. Language locale map (`LANGUAGE_LOCALE`)

16 languages (en, es, fr, de, it, pt, nl, hi, ja, zh, ar, ko, ru, pl, sv, tr) →
{gl, hl} for search locale. Non-English claims: the LLM is **mandated** to write both
`claim` and `explanation` in the transcript language; only verdict labels stay English.

## 11. Prompts (REWRITE — originals are empty in the committed `-ex` files)

`EVALUATE_PROMPT` and `GROUNDED_PROMPT` are empty strings in the public repo (the real
prompts were gitignored). We write fresh prompts in `mcp-servers/mcp-factcheck/prompts.py`
that capture the documented contract:
- EVALUATE: extract check-worthy factual claims from the window; exclude opinions,
  predictions, rhetoric, value judgments. Output JSON array of `{claim, verdict, speaker}`.
  Temporal grounding: "evaluate as of the recording date."
- GROUNDED: given a claim + web evidence, independently re-judge. Output a single
  `{claim, verdict, explanation, speaker}`. Evidence-only; ignore post-date info.

"Check-worthy" definition (from README): specific factual statements, statistics/numerical,
historical events, government actions/policies, scientific/medical claims, public records.
NOT: opinions, predictions/promises, rhetorical questions, value judgments, subjective descriptions.

## 12. Audio pipeline (reference — we replace Deepgram with local faster-whisper)

Original `offscreen.js`: `AudioContext({sampleRate:16000})` → `ScriptProcessor(4096,1,1)` →
Float32→Int16 PCM (`int16[i] = clamp(float32[i]*32768, -32768, 32767)`) → Deepgram WS
`wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&model=nova-2
&...&diarize=true`. Deepgram handled VAD + diarization + interim/final utterance boundaries.

**Our replacement (engine):** local Silero VAD gates faster-whisper. We lose Deepgram's
built-in diarization — for ambient single-speaker use this is fine; multi-speaker scenarios
will add WhisperX/pyannote later. The 16kHz Int16 PCM framing is preserved as the wire
format for all clients (browser, phone, desktop).

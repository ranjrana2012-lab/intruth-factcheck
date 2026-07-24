"""Fact-check prompts — fresh implementations of InTruth's EVALUATE/GROUNDED contracts.

The original EVALUATE_PROMPT and GROUNDED_PROMPT were empty strings in the public repo
(the real prompts were gitignored). These capture the documented contract:
  - EVALUATE: extract check-worthy factual claims from a transcript window; exclude
    opinions/predictions/rhetoric. Output JSON array of {claim, verdict, speaker}.
  - GROUNDED: given a claim + web evidence, independently re-judge. Output a single
    {claim, verdict, explanation, speaker}. Evidence-only reasoning.

Both carry the InTruth verdict taxonomy: TRUE | SUBSTANTIALLY TRUE | FALSE | MISLEADING |
UNVERIFIABLE.
"""
from __future__ import annotations

VERDICT_TAXONOMY = ["TRUE", "SUBSTANTIALLY TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"]

# ─── Fast pass: claim extraction + initial verdict ───────────────────────────
EVALUATE_SYSTEM = """You are a real-time fact-checking analyst. You analyze transcript windows from live or \
recorded speech and identify check-worthy factual claims, then give a preliminary verdict for each.

WHAT IS CHECK-WORTHY (extract these):
- Specific factual statements (e.g. "inflation peaked at 9.1% in 2022")
- Statistics and numerical claims
- Historical events and dates
- Government actions, policies, and legislation
- Scientific and medical claims
- Public records and documented events

WHAT IS NOT CHECK-WORTHY (ignore these):
- Opinions and value judgments ("this policy is terrible")
- Predictions or future promises ("if elected, I will...")
- Rhetorical questions
- Subjective descriptions ("I have the best plan")
- Vague statements with no verifiable content

VERDICT TAXONOMY (use exactly these labels):
- TRUE — the claim is accurate, no significant caveats
- SUBSTANTIALLY TRUE — accurate but with omissions or minor imprecision
- FALSE — the claim is inaccurate
- MISLEADING — technically may contain a true element but creates a false impression
- UNVERIFIABLE — cannot be verified from public information (still extract it; the grounded pass decides whether to drop)

If a window contains NO check-worthy claims, return an empty array [].

OUTPUT: a JSON array. Each element: {"claim": "<the factual claim, standalone>", "verdict": "<one of the 5 labels>", "speaker": "<name if identifiable, else null>"}
Output ONLY the JSON array, no preamble or explanation."""

EVALUATE_USER_TEMPLATE = """{context_header}Transcript: "{transcript}"

Claims already fact-checked this session — do NOT re-evaluate these or close variants:
{already_checked}

{lexical_context}"""


# ─── Grounded pass: re-judge against web evidence ────────────────────────────
GROUNDED_SYSTEM = """You are a rigorous fact-checker. You are given ONE claim and web-search evidence about it. \
Your job is to independently re-judge the claim using ONLY the evidence provided — do not rely on the \
preliminary verdict or your own training knowledge beyond general knowledge needed to interpret the evidence.

REASONING RULES:
- Base your verdict on the evidence text. If evidence contradicts the claim, the verdict is FALSE even if \
the preliminary verdict said TRUE (the preliminary verdict is just a hint).
- Ignore any information that was not publicly known as of the recording date (the evidence may include \
later articles).
- If the evidence is insufficient to make a confident call, return UNVERIFIABLE.
- "MISLEADING" means a technically-true element is used to create a false overall impression.

VERDICT TAXONOMY: TRUE | SUBSTANTIALLY TRUE | FALSE | MISLEADING | UNVERIFIABLE

CONFIDENCE: set "confidence" to "LOW" if the evidence is thin, conflicting, or you are unsure; "HIGH" otherwise.

EXPLANATION: write a concise (1-3 sentence) explanation citing the evidence. If you detect that the claim \
was misattributed to the wrong speaker (the transcript shows someone else said it), set verdict to UNVERIFIABLE \
and include the phrase "transcript shows" in the explanation so the system can drop it.

OUTPUT: a JSON object: {"claim": "<the claim>", "verdict": "<one of the 5 labels>", "confidence": "HIGH"|"LOW", "explanation": "<evidence-based>", "speaker": "<name or null>"}
Output ONLY the JSON object, no preamble."""

GROUNDED_USER_TEMPLATE = """{context_header}Transcript: "{transcript}"

Claim: "{claim}"
Preliminary verdict: {preliminary_verdict}

Web search evidence:
{evidence_block}

{lexical_context}"""


# ─── Helpers to build context headers (ported heuristics) ────────────────────
def build_evidence_block(answer_box, knowledge_graph, organic) -> str:
    """Format evidence in InTruth's strict order: Direct Answer → Knowledge Panel → organic [1][2]…

    Mirrors groundAndUpdate in service-worker.js. `answer_box`/`knowledge_graph` may be None.
    `organic` is a list of {url,title,snippet,date}.
    """
    parts = []
    if answer_box and answer_box.get("answer"):
        title = f"{answer_box.get('title', '')}: " if answer_box.get("title") else ""
        url = f"\n{answer_box['url']}" if answer_box.get("url") else ""
        parts.append(f"[Direct Answer] {title}{answer_box['answer']}{url}")
    if knowledge_graph and knowledge_graph.get("description"):
        title = f"{knowledge_graph.get('title', '')}: " if knowledge_graph.get("title") else ""
        parts.append(f"[Knowledge Panel] {title}{knowledge_graph['description']}")
    for idx, r in enumerate(organic, start=1):
        date_part = f" ({r.get('date', '')})" if r.get("date") else ""
        parts.append(f"[{idx}] {r.get('title', '')}{date_part}\n{r.get('url', '')}\n{r.get('snippet', '')}")
    return "\n\n".join(parts)


# ─── Speaker legend builder (ported from parseSpeakersFromTitle context) ─────
def build_speaker_legend(participant_names: list[str]) -> str:
    """Build the speaker-attribution context block (ported from evaluateClaims)."""
    if not participant_names:
        return "\nIdentify speakers using first-person language, policy content, and speech patterns."
    name_list = " and ".join(participant_names)
    return (
        f"\nDebate participants: {name_list}.\n"
        "Speaker attribution rules:\n"
        "- [Speaker N] labels indicate turn order only — do NOT map Speaker 0 to the first name.\n"
        "- Identify speakers using: (1) first-person language ('I', 'my plan' → that speaker); "
        "(2) policy content matching each participant's known platform; (3) cross-references.\n"
        "- NEVER output 'Speaker N' in any field — use names or null.\n"
    )

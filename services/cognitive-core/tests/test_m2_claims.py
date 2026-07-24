"""M2 claims-logic tests — pure logic (no LLM/retrieval calls)."""
import sys
from pathlib import Path

# Set up BOTH import roots before any test imports its modules.
# tests/ → cognitive-core → services → REPO_ROOT
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
MCP_PATH = REPO_ROOT / "mcp-servers" / "mcp-factcheck"
sys.path.insert(0, str(MCP_PATH))

# Pre-import the MCP-space modules so the `from retrieval...` / `from bias...` calls inside
# tests resolve against the path we just added.
import retrieval  # noqa: F401, E402
import bias  # noqa: F401, E402


def test_dedup_exact_and_overlap():
    from intruth_engine.claims import ClaimDeduplicator

    d = ClaimDeduplicator(ttl_ms=200_000, overlap_threshold=0.35)
    assert not d.is_duplicate("Inflation peaked at 9.1 percent in 2022")
    # exact restatement → dup
    assert d.is_duplicate("Inflation peaked at 9.1 percent in 2022")
    # different claim → not dup
    assert not d.is_duplicate("The bill passed the Senate unanimously")
    # close overlap → dup (shares most ≥4-char words)
    d2 = ClaimDeduplicator()
    assert not d2.is_duplicate("The unemployment rate fell below five percent")
    assert d2.is_duplicate("unemployment rate dropped below five percent")


def test_dedup_monetary_figure():
    from intruth_engine.claims import ClaimDeduplicator

    d = ClaimDeduplicator()
    assert not d.is_duplicate("The deficit reached $2.4 trillion this year")
    # same figure, different wording → dup via monetary guard
    assert d.is_duplicate("We spent $2.4 trillion on infrastructure")


def test_window_fires_on_size_and_speaker_change():
    from intruth_engine.claims import SentenceWindow

    # window_size=2 → fires every 2 sentences
    w = SentenceWindow(window_size=2, window_keep=6)
    snaps = []
    for s in ["First sentence.", "Second sentence.", "Third sentence."]:
        snaps += w.feed(s, speaker_id=0)
    assert len(snaps) == 1  # fired after 2nd
    assert "First sentence" in snaps[0].context_text

    # speaker change mid-window flushes early
    w2 = SentenceWindow(window_size=4, window_keep=8)
    snaps2 = []
    snaps2 += w2.feed("Alpha one.", speaker_id=0)
    snaps2 += w2.feed("Alpha two.", speaker_id=0)
    snaps2 += w2.feed("Bravo switching speaker now.", speaker_id=1)  # change → flush
    assert len(snaps2) == 1


def test_lexical_extraction():
    from intruth_engine.claims import extract_lexical, build_lexical_summary

    f = extract_lexical("I definitely believe this is the best plan, absolutely.")
    assert f.rates["certainty"] > 0  # 'definitely', 'absolutely', 'best'
    summary = build_lexical_summary(f)
    assert "certainty" in summary.lower()


def test_redact_strips_pii():
    from intruth_engine.pii import redact

    out = redact("Call me at john@example.com or 4111-1111-1111-1111")
    assert "<EMAIL>" in out
    assert "<CREDIT_CARD>" in out
    assert "john@example.com" not in out


def test_source_filter_blocks_partisan():
    from retrieval.filter import is_blocked, filter_results, OrganicResult

    assert is_blocked("https://www.breitbart.com/politics/story")
    assert is_blocked("https://reddit.com/r/news")
    assert not is_blocked("https://www.reuters.com/world/us/story")
    results = [
        OrganicResult(url="https://breitbart.com/x", title="x", snippet=""),
        OrganicResult(url="https://reuters.com/y", title="y", snippet=""),
    ]
    filtered = filter_results(results, max_results=4)
    assert len(filtered) == 1
    assert "reuters" in filtered[0].url


def test_evidence_block_ordering():
    from intruth_engine.claims.prompts import build_evidence_block

    block = build_evidence_block(
        answer_box={"answer": "Yes", "title": "T", "url": "u"},
        knowledge_graph={"description": "Desc", "title": "K"},
        organic=[{"url": "o1", "title": "O1", "snippet": "s1", "date": "2024-01-01"}],
    )
    assert block.index("[Direct Answer]") < block.index("[Knowledge Panel]")
    assert block.index("[Knowledge Panel]") < block.index("[1]")


def test_json_extraction():
    from intruth_engine.llm import extract_json_array, extract_json_object

    arr = extract_json_array("blah ```json\n[{\"claim\": \"x\", \"verdict\": \"TRUE\"}]\n``` end")
    assert arr == [{"claim": "x", "verdict": "TRUE"}]
    obj = extract_json_object("text {\"claim\":\"y\",\"verdict\":\"FALSE\"} tail")
    assert obj["verdict"] == "FALSE"


def test_bias_lookup():
    from bias import lookup_bias, annotate_sources

    b = lookup_bias("https://www.reuters.com/article/123")
    assert b and b["credibility"] == "high"
    annotated = annotate_sources(["https://reuters.com/x", "https://example.com/y"])
    assert any(a["domain"] == "reuters.com" for a in annotated)

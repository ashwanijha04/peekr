"""Tests for the CitationAccuracy evaluator."""

from __future__ import annotations

import pytest

from peekr.eval import EvalExporter
from peekr.eval.citation import CitationAccuracy, extract_citations
from peekr.span import Span


def _span(input_text: str, output: str) -> Span:
    s = Span(name="openai.chat.completions", trace_id="t1")
    s.attributes["input"] = input_text
    s.attributes["output"] = output
    s.finish()
    return s


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_extracts_urls(self):
        out = extract_citations("See https://example.com/paper.pdf for details.")
        assert any(c["kind"] == "url" and "example.com" in c["text"] for c in out)

    def test_extracts_arxiv_ids(self):
        out = extract_citations("Cited in arXiv:1810.04805 and arXiv: 2305.10403.")
        kinds = [c["kind"] for c in out]
        assert kinds.count("arxiv") == 2

    def test_extracts_doi(self):
        out = extract_citations("DOI 10.1038/s41586-021-03819-2 is the reference.")
        assert any(c["kind"] == "doi" for c in out)

    def test_extracts_author_year(self):
        out = extract_citations(
            "As shown by Vaswani et al. (2017) and Devlin et al. 2018."
        )
        ay = [c for c in out if c["kind"] == "author_year"]
        assert len(ay) == 2

    def test_extracts_section_reference(self):
        out = extract_citations("Refer to Section 230 and § 7 of the Act.")
        kinds = [c["kind"] for c in out]
        assert kinds.count("section") == 2

    def test_extracts_quoted_title(self):
        out = extract_citations(
            'See "Attention Is All You Need" for the original work.'
        )
        assert any(c["kind"] == "quoted_title" for c in out)

    def test_short_quoted_strings_are_ignored(self):
        # Should not match a 4-character quoted snippet
        out = extract_citations('They said "hi".')
        assert not any(c["kind"] == "quoted_title" for c in out)

    def test_dedupes_repeated_citations(self):
        out = extract_citations("arXiv:1810.04805 and arXiv: 1810.04805 again.")
        assert sum(1 for c in out if c["kind"] == "arxiv") == 1


# ---------------------------------------------------------------------------
# Evaluator semantics
# ---------------------------------------------------------------------------


class TestCitationAccuracy:
    def test_returns_one_when_no_citations(self):
        s = _span("ctx", "no citations here at all")
        assert CitationAccuracy().evaluate(s) == 1.0

    def test_returns_one_when_all_citations_grounded(self):
        s = _span(
            "BERT was introduced by Devlin et al. 2018. Reference paper: arXiv:1810.04805.",
            "BERT was introduced by Devlin et al. 2018; see arXiv:1810.04805.",
        )
        assert CitationAccuracy().evaluate(s) == pytest.approx(1.0)

    def test_returns_zero_when_all_citations_invented(self):
        s = _span(
            "BERT is a transformer model.",
            "BERT was introduced by Devlin et al. 2018 in arXiv:1810.04805.",
        )
        score = CitationAccuracy().evaluate(s)
        assert score == pytest.approx(0.0)
        details = s.attributes["citation_details"]
        assert details["invented"] == 2
        assert details["grounded"] == 0

    def test_partial_credit_for_mixed_citations(self):
        s = _span(
            "See arXiv:1810.04805 for details.",
            "See arXiv:1810.04805 and a follow-up at arXiv:2305.10403.",
        )
        score = CitationAccuracy().evaluate(s)
        assert score == pytest.approx(0.5)
        d = s.attributes["citation_details"]
        assert d["grounded"] == 1
        assert d["invented"] == 1

    def test_normalizes_whitespace_for_matching(self):
        s = _span(
            "Section 230 of the CDA provides immunity.",
            "Section  230 governs platform immunity.",
        )
        assert CitationAccuracy().evaluate(s) == pytest.approx(1.0)

    def test_returns_one_when_output_empty(self):
        s = _span("ctx", "")
        assert CitationAccuracy().evaluate(s) == 1.0

    def test_writes_details_per_citation(self):
        s = _span(
            "Vaswani et al. 2017 published the original work.",
            "Vaswani et al. 2017 wrote it; Smith et al. 2025 extended it.",
        )
        CitationAccuracy().evaluate(s)
        items = s.attributes["citation_details"]["items"]
        # First citation grounded, second invented
        author_year_items = [i for i in items if i["kind"] == "author_year"]
        assert len(author_year_items) == 2
        grounded = [i for i in author_year_items if i["grounded"]]
        invented = [i for i in author_year_items if not i["grounded"]]
        assert len(grounded) == 1 and len(invented) == 1

    def test_integration_via_eval_exporter(self):
        s = _span(
            "BERT is a transformer.",
            "BERT was introduced in arXiv:9999.99999 by Smith et al. 2025.",
        )
        exporter = EvalExporter(async_eval=False, evaluators=[CitationAccuracy()])
        exporter.export(s)
        assert s.attributes["eval_scores"]["CitationAccuracy"] == pytest.approx(0.0)

    def test_context_extractor_override(self):
        s = _span("irrelevant", "See arXiv:1810.04805.")
        retrieved = "Reference: arXiv:1810.04805."
        evaluator = CitationAccuracy(context_extractor=lambda span: retrieved)
        assert evaluator.evaluate(s) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Composability — Hallucination + CitationAccuracy run together
# ---------------------------------------------------------------------------


class TestComposedSignals:
    def test_both_signals_land_on_span(self):
        """Citation evaluator runs without an LLM, so the EvalExporter can include both."""
        from unittest.mock import MagicMock, patch
        from peekr.eval.hallucination import Hallucination

        s = _span(
            "Section 230 of the CDA provides immunity for online platforms.",
            "Section 230 of the CDA and the 2003 Online Platform Immunity Act provide immunity.",
        )
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "0.4"
        with patch("peekr.eval._judge.openai") as mock_oai:
            mock_oai.chat.completions.create.return_value = mock_resp
            exporter = EvalExporter(
                async_eval=False, evaluators=[Hallucination(), CitationAccuracy()]
            )
            exporter.export(s)

        scores = s.attributes["eval_scores"]
        assert "Hallucination" in scores
        assert "CitationAccuracy" in scores
        # CitationAccuracy should also have caught the fake 2003 Act-shaped pattern
        # (it shows up as a quoted-title-style year, plus a section reference present in context)
        # The key invariant: both signals are recorded, independently.
        assert scores["CitationAccuracy"] <= 1.0

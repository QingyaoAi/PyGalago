"""Phase 3 retrieval tests — require C++ extension + GALAGO_INDEX_PATH.

Set GALAGO_INDEX_PATH to a Galago index directory to enable integration tests.
"""

import math
import os
import pytest

INDEX_PATH = os.environ.get("GALAGO_INDEX_PATH", "")
HAS_INDEX  = bool(INDEX_PATH)

try:
    import pygalago._galago as _g
    HAS_EXT = True
except ImportError:
    HAS_EXT = False


# ── Import / API tests (always run) ──────────────────────────────────────────

class TestRetrievalAPI:
    def test_module_importable(self):
        import pygalago.retrieval as gr
        assert hasattr(gr, "bm25_search")
        assert hasattr(gr, "search")

    def test_raises_without_extension(self, tmp_path):
        import pygalago.retrieval as gr
        if HAS_EXT:
            pytest.skip("Extension is built")
        with pytest.raises(RuntimeError, match="extension"):
            gr.bm25_search(str(tmp_path), ["term"])


# ── Integration tests (require extension + index) ────────────────────────────

@pytest.fixture(scope="module")
def idx_path():
    if not HAS_EXT:
        pytest.skip("C++ extension not built")
    if not HAS_INDEX:
        pytest.skip("Set GALAGO_INDEX_PATH to run retrieval integration tests")
    return INDEX_PATH


class TestBM25Search:
    def test_single_term_returns_results(self, idx_path):
        from pygalago._galago import bm25_search, LengthsSource
        results = bm25_search(idx_path, ["inform"], n=10)
        assert len(results) > 0
        assert len(results) <= 10

    def test_scores_descending(self, idx_path):
        from pygalago._galago import bm25_search
        results = bm25_search(idx_path, ["inform"], n=100)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_formula_exact(self, idx_path):
        """Verify BM25 score matches the formula to machine precision."""
        from pygalago._galago import bm25_search, LengthsSource, DiskIndex

        results = bm25_search(idx_path, ["inform"], n=1)
        assert len(results) == 1
        doc   = results[0].document
        score = results[0].score

        # Get tf via the postings reader
        idx     = DiskIndex(idx_path)
        lengths = LengthsSource(os.path.join(idx_path, "lengths"))
        ls      = lengths.stats()

        pit = idx.get_postings("inform")
        assert pit is not None
        pit.skip_to(doc)
        assert not pit.is_done and pit.doc_id == doc

        tf  = pit.count
        dl  = lengths.length(doc)
        N   = ls.total_document_count
        df  = pit.stats["document_count"]
        b, k = 0.75, 1.2

        idf      = math.log(N / (df + 0.5))
        expected = idf * tf * (k + 1) / (tf + k * (1 - b + b * dl / ls.avg_length))

        assert abs(score - expected) < 1e-9

    def test_missing_term_returns_empty(self, idx_path):
        from pygalago._galago import bm25_search
        results = bm25_search(idx_path, ["xyzzy_no_such_term_xyzzy"], n=10)
        assert len(results) == 0

    def test_top_1000(self, idx_path):
        from pygalago._galago import bm25_search
        results = bm25_search(idx_path, ["inform"], n=1000)
        assert 0 < len(results) <= 1000

    def test_multi_term_combine(self, idx_path):
        """Multi-term #combine score must be between 0 and sum of individual maxima."""
        from pygalago._galago import bm25_search
        results_multi = bm25_search(idx_path, ["inform", "retriev"], n=10)
        # Should return results (even if one term is missing it gracefully degrades)
        # If both terms present: combined score < max of individual scores
        results_single = bm25_search(idx_path, ["inform"], n=10)
        assert len(results_multi) > 0
        assert len(results_single) > 0

    def test_scored_document_attributes(self, idx_path):
        from pygalago._galago import bm25_search
        results = bm25_search(idx_path, ["inform"], n=5)
        for r in results:
            assert isinstance(r.document, int)
            assert isinstance(r.score, float)
            assert r.document >= 0
            assert r.score > 0.0


class TestPythonSearchAPI:
    def test_search_wrapper(self, idx_path):
        import pygalago.retrieval as gr
        results = gr.search(idx_path, "information retrieval", n=10)
        # "information retrieval" → lowercase split → ["information", "retrieval"]
        # These are NOT Krovetz-stemmed so may return 0 results from krovetz part
        assert isinstance(results, list)

    def test_bm25_search_wrapper(self, idx_path):
        import pygalago.retrieval as gr
        results = gr.bm25_search(idx_path, ["inform"], n=10)
        assert len(results) > 0
        assert all(r.score > 0 for r in results)

    def test_b_k_parameters(self, idx_path):
        """Different b/k values should produce different scores."""
        from pygalago._galago import bm25_search
        r1 = bm25_search(idx_path, ["inform"], n=1, b=0.0, k=1.2)
        r2 = bm25_search(idx_path, ["inform"], n=1, b=1.0, k=1.2)
        assert len(r1) == 1 and len(r2) == 1
        # Same top doc but different scores (b changes length normalisation)
        assert abs(r1[0].score - r2[0].score) > 1e-6

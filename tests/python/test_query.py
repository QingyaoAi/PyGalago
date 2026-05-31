"""Tests for Phase 4: query parsing, traversals, and the Retrieval pipeline.

Parser + traversal tests always run (no C++ extension needed).
Integration tests require extension + GALAGO_INDEX_PATH (auto-discovered).
"""

import os
import pytest
import warnings

INDEX_PATH = os.environ.get("GALAGO_INDEX_PATH", "")
HAS_INDEX  = bool(INDEX_PATH)

try:
    import pygalago._galago as _g
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

from pygalago.query import parse, find_query_terms, Node
from pygalago.query.iterator_builder import node_to_weighted_terms
from pygalago.query.traversals import (
    FullDependenceTraversal,
    PartAssignerTraversal,
)


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:
    def test_single_term(self):
        n = parse("information")
        assert n.operator == "text"
        assert n.default_parameter == "information"

    def test_two_terms_implicit_combine(self):
        n = parse("information retrieval")
        assert n.operator == "combine"
        assert len(n.children) == 2
        assert n.children[0].default_parameter == "information"
        assert n.children[1].default_parameter == "retrieval"

    def test_explicit_combine(self):
        n = parse("#combine(information retrieval)")
        assert n.operator == "combine"
        assert len(n.children) == 2

    def test_weighted_combine(self):
        n = parse("#combine:0=0.7:1=0.3(information retrieval)")
        assert n.operator == "combine"
        assert abs(n.params["0"] - 0.7) < 1e-9
        assert abs(n.params["1"] - 0.3) < 1e-9

    def test_ordered_window(self):
        n = parse("#od:2(information retrieval)")
        assert n.operator == "od"
        assert n.params.get("default") == 2 or n.params.get("default") == "2"
        assert len(n.children) == 2

    def test_fdm_operator(self):
        n = parse("#fdm(information retrieval)")
        assert n.operator == "fdm"
        assert len(n.children) == 2

    def test_nested_combine(self):
        n = parse("#combine(#od:1(t1 t2) #uw:8(t1 t2))")
        assert n.operator == "combine"
        assert n.children[0].operator == "od"
        assert n.children[1].operator == "uw"

    def test_quoted_term(self):
        n = parse('"information retrieval"')
        # A quoted phrase is a single text node with the whole phrase
        assert n.operator == "text"
        assert "information retrieval" in n.default_parameter

    def test_empty_query(self):
        n = parse("")
        assert n.operator == "text"

    def test_roundtrip(self):
        queries = [
            "information",
            "#combine(information retrieval)",
            "#fdm(information retrieval)",
        ]
        for q in queries:
            n = parse(q)
            assert str(n)  # must produce non-empty string

    def test_find_query_terms(self):
        n = parse("information retrieval")
        terms = find_query_terms(n)
        assert "information" in terms
        assert "retrieval" in terms


# ── FullDependenceTraversal tests ─────────────────────────────────────────────

class TestFullDependenceTraversal:
    def setup_method(self):
        self.fdm = FullDependenceTraversal()

    def test_single_term_no_expand(self):
        n = parse("#fdm(information)")
        result = self.fdm.traverse(n)
        # Single term → just the unigram combine
        assert result.operator == "combine"
        assert len(result.children) == 1

    def test_two_terms_expansion(self):
        n = parse("#fdm(t1 t2)")
        result = self.fdm.traverse(n)
        # Should produce: #combine:0=0.8:1=0.15:2=0.05(unigrams od uw)
        assert result.operator == "combine"
        assert abs(result.params.get("0", 0) - 0.8)  < 1e-9
        assert abs(result.params.get("1", 0) - 0.15) < 1e-9
        assert abs(result.params.get("2", 0) - 0.05) < 1e-9
        assert len(result.children) == 3  # unigrams, ordered, unordered

    def test_non_fdm_passthrough(self):
        n = parse("#combine(t1 t2)")
        result = self.fdm.traverse(n)
        assert result.operator == "combine"
        assert len(result.children) == 2

    def test_custom_weights(self):
        fdm = FullDependenceTraversal({"uniw": 0.6, "odw": 0.3, "uww": 0.1})
        n = parse("#fdm(t1 t2)")
        result = fdm.traverse(n)
        assert abs(result.params["0"] - 0.6) < 1e-9
        assert abs(result.params["1"] - 0.3) < 1e-9


# ── PartAssignerTraversal tests ───────────────────────────────────────────────

class TestPartAssignerTraversal:
    def test_assigns_part(self):
        n = parse("information retrieval")
        pa = PartAssignerTraversal("postings.krovetz")
        result = pa.traverse(n)
        for child in result.children:
            assert child.params.get("part") == "postings.krovetz"

    def test_respects_existing_part(self):
        n = Node("text", {"default": "info", "part": "postings"}, [])
        pa = PartAssignerTraversal("postings.krovetz")
        result = pa.traverse(n)
        assert result.params["part"] == "postings"  # not overwritten


# ── Iterator builder tests ────────────────────────────────────────────────────

class TestIteratorBuilder:
    def test_single_leaf(self):
        n = Node.text("information")
        terms = node_to_weighted_terms(n)
        assert len(terms) == 1
        assert terms[0][0] == "information"
        assert abs(terms[0][1] - 1.0) < 1e-9

    def test_combine_uniform(self):
        n = parse("information retrieval")
        terms = node_to_weighted_terms(n)
        assert len(terms) == 2
        total = sum(w for _, w in terms)
        assert abs(total - 1.0) < 1e-9
        # Uniform weights
        for _, w in terms:
            assert abs(w - 0.5) < 1e-9

    def test_combine_explicit_weights(self):
        n = parse("#combine:0=0.7:1=0.3(information retrieval)")
        terms = node_to_weighted_terms(n)
        d = dict(terms)
        assert abs(d["information"] - 0.7) < 1e-9
        assert abs(d["retrieval"]   - 0.3) < 1e-9

    def test_fdm_proxy_skipped_with_warning(self):
        fdm = FullDependenceTraversal()
        n = parse("#fdm(t1 t2)")
        n = fdm.traverse(n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            terms = node_to_weighted_terms(n)
            # Should warn about ordered/unordered being skipped
            assert any("proximity" in str(x.message).lower()
                       or "ordered" in str(x.message).lower()
                       or "positional" in str(x.message).lower()
                       for x in w)
        # After skipping od+uw, only unigrams remain — re-normalised to 1.0
        total = sum(w for _, w in terms)
        assert abs(total - 1.0) < 1e-9

    def test_deduplication(self):
        # Same term in two branches → weights summed, then normalised
        n = Node("combine", {"0": 0.5, "1": 0.5}, [
            Node.text("info"),
            Node.text("info"),
        ])
        terms = node_to_weighted_terms(n)
        assert len(terms) == 1
        assert terms[0][0] == "info"
        assert abs(terms[0][1] - 1.0) < 1e-9


# ── Full Retrieval pipeline integration tests ─────────────────────────────────

@pytest.fixture(scope="module")
def retrieval():
    if not HAS_EXT:
        pytest.skip("C++ extension not built")
    if not HAS_INDEX:
        pytest.skip("Set GALAGO_INDEX_PATH to run Retrieval integration tests")
    from pygalago.retrieval import Retrieval
    return Retrieval(INDEX_PATH)


class TestRetrieval:
    def test_bare_word_query(self, retrieval):
        results = retrieval.search("information", n=10)
        assert len(results) > 0
        names, scores = zip(*results)
        assert all(isinstance(nm, str) and len(nm) > 0 for nm in names)
        assert list(scores) == sorted(scores, reverse=True)

    def test_multi_word_query(self, retrieval):
        results = retrieval.search("information retrieval", n=10)
        assert len(results) > 0
        _, scores = zip(*results)
        assert list(scores) == sorted(scores, reverse=True)

    def test_explicit_combine(self, retrieval):
        r1 = retrieval.search("information retrieval", n=10)
        r2 = retrieval.search("#combine(information retrieval)", n=10)
        # Uniform combine should give same ranking
        n1 = [nm for nm, _ in r1]
        n2 = [nm for nm, _ in r2]
        assert n1 == n2

    def test_weighted_combine_changes_ranking(self, retrieval):
        # Extreme weight on one term changes order (usually)
        r1 = retrieval.search("information retrieval", n=100)
        r2 = retrieval.search(
            "#combine:0=0.99:1=0.01(information retrieval)", n=100)
        names1 = [nm for nm, _ in r1]
        names2 = [nm for nm, _ in r2]
        # Top-100 sets likely differ with extreme weighting
        assert set(names1) != set(names2) or names1 != names2

    def test_fdm_query(self, retrieval):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = retrieval.search("#fdm(information retrieval)", n=10)
        # Even with od/uw skipped, should return results
        assert len(results) > 0

    def test_missing_term_graceful(self, retrieval):
        results = retrieval.search("xyzzynosuchterm", n=10)
        assert results == []

    def test_search_scored_returns_scored_documents(self, retrieval):
        results = retrieval.search_scored("information", n=5)
        assert len(results) > 0
        r = results[0]
        assert hasattr(r, "document") and hasattr(r, "score")

    def test_explain(self, retrieval):
        info = retrieval.explain("information retrieval")
        assert "raw_tree" in info
        assert "processed_tree" in info
        assert "weighted_terms" in info
        assert len(info["weighted_terms"]) > 0

    def test_scores_match_bm25_formula(self, retrieval):
        """Score from Retrieval.search must equal manual BM25 computation."""
        import math
        results = retrieval.search_scored("information", n=1)
        assert results
        doc   = results[0].document
        score = results[0].score

        ls = retrieval._ls
        pr = retrieval._pr
        stats = pr.get_stats("information")
        assert stats is not None

        length = retrieval._lengths.length(doc)
        pit = retrieval._index.get_postings("information", retrieval.part)
        pit.skip_to(doc)
        assert not pit.is_done and pit.doc_id == doc
        tf = pit.count

        N   = ls.total_document_count
        df  = stats["document_count"]
        avg = ls.avg_length
        b, k = retrieval.b, retrieval.k
        idf  = math.log(N / (df + 0.5))
        expected = idf * tf * (k + 1) / (tf + k * (1 - b + b * length / avg))

        assert abs(score - expected) < 1e-9

    def test_m3_milestone_structured_query(self, retrieval):
        """Milestone M3: structured #combine query matches bare word ranking."""
        r_bare    = retrieval.search("information retrieval", n=100)
        r_combine = retrieval.search("#combine(information retrieval)", n=100)
        # Structured #combine with uniform weights must produce identical ranking
        assert r_bare == r_combine

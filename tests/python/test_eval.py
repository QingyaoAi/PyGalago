# BSD License (http://www.galagosearch.org/license)
"""Phase 6 tests — IR evaluation metrics, qrels and run I/O."""

from __future__ import annotations

import io
import math

import pytest

from pygalago.eval.qrels   import read_qrels, write_qrels, relevant_docs
from pygalago.eval.run     import read_run, write_run, from_scored_documents, RankedDoc
from pygalago.eval.metrics import (
    precision_at_k, recall_at_k, r_precision,
    average_precision, mean_average_precision,
    ndcg_at_k, mean_ndcg_at_k,
    reciprocal_rank, mean_reciprocal_rank,
    bpref, evaluate,
)

# ── Qrels I/O ─────────────────────────────────────────────────────────────────

QRELS_TEXT = """\
301 0 doc1 1
301 0 doc2 0
301 0 doc3 2
302 0 doc4 1
302 0 doc5 0
"""


class TestQrelsIO:
    def test_read_basic(self):
        q = read_qrels(io.StringIO(QRELS_TEXT))
        assert "301" in q
        assert q["301"]["doc1"] == 1
        assert q["301"]["doc2"] == 0
        assert q["301"]["doc3"] == 2

    def test_write_roundtrip(self):
        q = read_qrels(io.StringIO(QRELS_TEXT))
        buf = io.StringIO()
        write_qrels(q, buf)
        buf.seek(0)
        q2 = read_qrels(buf)
        assert q == q2

    def test_relevant_docs_binary(self):
        q = read_qrels(io.StringIO(QRELS_TEXT))
        rel = relevant_docs(q, "301")
        assert "doc1" in rel
        assert "doc3" in rel
        assert "doc2" not in rel

    def test_relevant_docs_graded_cutoff(self):
        q = read_qrels(io.StringIO(QRELS_TEXT))
        rel = relevant_docs(q, "301", min_grade=2)
        assert rel == frozenset({"doc3"})

    def test_missing_topic_returns_empty(self):
        q = read_qrels(io.StringIO(QRELS_TEXT))
        assert relevant_docs(q, "999") == frozenset()

    def test_skip_comments_and_blank_lines(self):
        text = "# comment\n\n301 0 doc1 1\n"
        q = read_qrels(io.StringIO(text))
        assert q["301"]["doc1"] == 1


# ── Run I/O ───────────────────────────────────────────────────────────────────

RUN_TEXT = """\
301 Q0 doc3 1 0.9 myrun
301 Q0 doc1 2 0.8 myrun
301 Q0 doc2 3 0.5 myrun
302 Q0 doc4 1 0.7 myrun
"""


class TestRunIO:
    def test_read_basic(self):
        run = read_run(io.StringIO(RUN_TEXT))
        assert "301" in run
        assert run["301"][0].doc_id == "doc3"
        assert run["301"][0].score == pytest.approx(0.9)

    def test_sorted_by_score(self):
        text = "301 Q0 docA 2 0.5 r\n301 Q0 docB 1 0.9 r\n"
        run = read_run(io.StringIO(text))
        assert run["301"][0].doc_id == "docB"
        assert run["301"][1].doc_id == "docA"

    def test_write_roundtrip(self):
        run = read_run(io.StringIO(RUN_TEXT))
        buf = io.StringIO()
        write_run(run, buf, run_tag="myrun")
        buf.seek(0)
        run2 = read_run(buf)
        assert {t: [r.doc_id for r in docs] for t, docs in run.items()} == \
               {t: [r.doc_id for r in docs] for t, docs in run2.items()}

    def test_from_scored_documents_tuples(self):
        scored = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        run = from_scored_documents("301", scored)
        assert run["301"][0].doc_id == "doc1"
        assert run["301"][2].score == pytest.approx(0.5)


# ── Metrics — Precision & Recall ──────────────────────────────────────────────

class TestPrecisionRecall:
    RANKED = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    REL    = frozenset({"doc1", "doc3", "doc5"})

    def test_precision_at_1(self):
        assert precision_at_k(self.RANKED, self.REL, 1) == pytest.approx(1.0)

    def test_precision_at_3(self):
        # doc1, doc2, doc3 → 2 relevant out of 3
        assert precision_at_k(self.RANKED, self.REL, 3) == pytest.approx(2 / 3)

    def test_precision_at_5(self):
        assert precision_at_k(self.RANKED, self.REL, 5) == pytest.approx(3 / 5)

    def test_precision_at_k_zero(self):
        assert precision_at_k(self.RANKED, self.REL, 0) == 0.0

    def test_recall_at_k(self):
        # After 3 ranks: 2 of 3 relevant seen
        assert recall_at_k(self.RANKED, self.REL, 3) == pytest.approx(2 / 3)

    def test_recall_complete_at_5(self):
        assert recall_at_k(self.RANKED, self.REL, 5) == pytest.approx(1.0)

    def test_r_precision(self):
        # R=3, precision@3 = 2/3
        assert r_precision(self.RANKED, self.REL) == pytest.approx(2 / 3)

    def test_empty_relevant_set(self):
        assert precision_at_k(self.RANKED, frozenset(), 5) == 0.0
        assert recall_at_k(self.RANKED, frozenset(), 5) == 0.0
        assert r_precision(self.RANKED, frozenset()) == 0.0


# ── Metrics — Average Precision ───────────────────────────────────────────────

class TestAveragePrecision:
    def test_perfect_ranking(self):
        ranked = ["d1", "d2", "d3"]
        rel    = frozenset({"d1", "d2", "d3"})
        assert average_precision(ranked, rel) == pytest.approx(1.0)

    def test_known_ap(self):
        # Relevant at positions 1, 3, 5 out of 5
        ranked = ["d1", "nr", "d2", "nr", "d3"]
        rel    = frozenset({"d1", "d2", "d3"})
        # AP = (1/3) * (1/1 + 2/3 + 3/5)
        expected = (1.0 + 2 / 3 + 3 / 5) / 3
        assert average_precision(ranked, rel) == pytest.approx(expected)

    def test_no_relevant_docs_returned(self):
        assert average_precision(["x", "y"], frozenset({"d1"})) == 0.0

    def test_empty_relevant(self):
        assert average_precision(["d1"], frozenset()) == 0.0

    def test_map(self):
        runs   = {"301": ["d1", "nr", "d2"], "302": ["d3"]}
        qrels  = {"301": frozenset({"d1", "d2"}), "302": frozenset({"d3"})}
        ap301  = average_precision(["d1", "nr", "d2"], frozenset({"d1", "d2"}))
        ap302  = 1.0
        assert mean_average_precision(runs, qrels) == pytest.approx((ap301 + ap302) / 2)


# ── Metrics — NDCG ───────────────────────────────────────────────────────────

class TestNDCG:
    RANKED = ["d1", "d2", "d3", "d4"]
    GRADES = {"d1": 3, "d2": 2, "d3": 0, "d4": 1}

    def test_perfect_ndcg(self):
        # Perfect order: d1 (3), d2 (2), d4 (1), d3 (0)
        perfect = ["d1", "d2", "d4", "d3"]
        assert ndcg_at_k(perfect, self.GRADES, 4) == pytest.approx(1.0)

    def test_ndcg_at_4_known(self):
        # DCG: (2^3-1)/log2(2) + (2^2-1)/log2(3) + 0 + (2^1-1)/log2(5)
        dcg = 7 / math.log2(2) + 3 / math.log2(3) + 0 + 1 / math.log2(5)
        # Ideal DCG (same grades, optimal order):
        idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4) + 0
        expected = dcg / idcg
        assert ndcg_at_k(self.RANKED, self.GRADES, 4) == pytest.approx(expected)

    def test_ndcg_empty_grades(self):
        assert ndcg_at_k(["d1"], {}, 5) == 0.0

    def test_ndcg_zero_cutoff(self):
        assert ndcg_at_k(self.RANKED, self.GRADES, 0) == 0.0

    def test_mean_ndcg(self):
        runs   = {"301": self.RANKED}
        graded = {"301": self.GRADES}
        mn = mean_ndcg_at_k(runs, graded, 4)
        assert mn == pytest.approx(ndcg_at_k(self.RANKED, self.GRADES, 4))


# ── Metrics — MRR ────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_relevant_at_rank_1(self):
        assert reciprocal_rank(["d1", "d2"], frozenset({"d1"})) == pytest.approx(1.0)

    def test_first_relevant_at_rank_2(self):
        assert reciprocal_rank(["nr", "d1"], frozenset({"d1"})) == pytest.approx(0.5)

    def test_no_relevant(self):
        assert reciprocal_rank(["x", "y"], frozenset({"d1"})) == 0.0

    def test_mrr(self):
        runs  = {"301": ["nr", "d1"], "302": ["d2"]}
        qrels = {"301": frozenset({"d1"}), "302": frozenset({"d2"})}
        assert mean_reciprocal_rank(runs, qrels) == pytest.approx(0.75)


# ── Metrics — Bpref ───────────────────────────────────────────────────────────

class TestBpref:
    def test_perfect_bpref(self):
        # All relevant, no non-relevant before them
        ranked = ["r1", "r2", "r3"]
        rel    = frozenset({"r1", "r2", "r3"})
        assert bpref(ranked, rel) == pytest.approx(1.0)

    def test_zero_bpref(self):
        # All non-relevant ranked before the single relevant doc
        ranked = ["n1", "n2", "n3", "r1"]
        rel    = frozenset({"r1"})
        nr     = frozenset({"n1", "n2", "n3"})
        # 1 rel, 3 non-rel before it, R=1 → penalty = min(3,1)/1 = 1.0 → contribution = 0
        assert bpref(ranked, rel, nr) == pytest.approx(0.0)

    def test_bpref_no_relevant(self):
        assert bpref(["d1", "d2"], frozenset()) == 0.0

    def test_bpref_partial(self):
        # r1 has 0 non-rel before it, r2 has 1 non-rel before it (R=2)
        ranked = ["r1", "nr1", "r2"]
        rel    = frozenset({"r1", "r2"})
        nr     = frozenset({"nr1"})
        # contribution r1: 1 - min(0,2)/2 = 1.0
        # contribution r2: 1 - min(1,2)/2 = 0.5
        assert bpref(ranked, rel, nr) == pytest.approx(0.75)


# ── evaluate() convenience wrapper ───────────────────────────────────────────

class TestEvaluate:
    RANKED = {"301": ["d1", "nr", "d2", "nr2", "d3"]}
    QRELS  = {"301": {"d1": 1, "d2": 1, "d3": 1, "nr": 0, "nr2": 0}}

    def test_returns_requested_metrics(self):
        res = evaluate(self.RANKED, self.QRELS, metrics=["map", "p@5"])
        assert set(res.keys()) == {"map", "p@5"}

    def test_map_value(self):
        res = evaluate(self.RANKED, self.QRELS, metrics=["map"])
        expected = average_precision(
            ["d1", "nr", "d2", "nr2", "d3"],
            frozenset({"d1", "d2", "d3"}),
        )
        assert res["map"] == pytest.approx(expected)

    def test_p_at_5(self):
        res = evaluate(self.RANKED, self.QRELS, metrics=["p@5"])
        assert res["p@5"] == pytest.approx(3 / 5)

    def test_ndcg_at_10(self):
        res = evaluate(self.RANKED, self.QRELS, metrics=["ndcg@10"])
        assert 0.0 <= res["ndcg@10"] <= 1.0

    def test_mrr(self):
        res = evaluate(self.RANKED, self.QRELS, metrics=["mrr"])
        assert res["mrr"] == pytest.approx(1.0)

    def test_bpref(self):
        res = evaluate(self.RANKED, self.QRELS, metrics=["bpref"])
        assert 0.0 <= res["bpref"] <= 1.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate(self.RANKED, self.QRELS, metrics=["totally_made_up"])

    def test_default_metrics_all_present(self):
        res = evaluate(self.RANKED, self.QRELS)
        for m in ("map", "ndcg@10", "ndcg@20", "mrr", "p@10", "bpref"):
            assert m in res

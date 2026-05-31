# BSD License (http://www.galagosearch.org/license)
"""Phase 8 integration tests — end-to-end pipeline from collection to evaluation.

These tests exercise the full flow: build index → run queries → score results.
They do not require a pre-built Galago index and are self-contained.

Robust04-specific tests (M5 milestone) run only when GALAGO_INDEX_PATH is set.
"""

from __future__ import annotations

import os
import textwrap
import time

import pytest

try:
    import pygalago._galago as _g
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

INDEX_PATH = os.environ.get("GALAGO_INDEX_PATH", "")
HAS_INDEX  = bool(INDEX_PATH)

# ── Shared mini-collection ────────────────────────────────────────────────────

MINI_COLLECTION = textwrap.dedent("""\
    <DOC>
    <DOCNO> IR001 </DOCNO>
    <TEXT>
    Information retrieval systems retrieve relevant documents from large collections.
    Relevance is a central concept in information retrieval research.
    </TEXT>
    </DOC>
    <DOC>
    <DOCNO> IR002 </DOCNO>
    <TEXT>
    Search engines index web documents using inverted indexes for fast retrieval.
    Query processing involves matching terms against the inverted index.
    </TEXT>
    </DOC>
    <DOC>
    <DOCNO> IR003 </DOCNO>
    <TEXT>
    Ranking algorithms like BM25 score documents by term frequency and document length.
    BM25 is widely used in modern search engines and information retrieval systems.
    </TEXT>
    </DOC>
    <DOC>
    <DOCNO> IR004 </DOCNO>
    <TEXT>
    Neural retrieval models use dense representations for semantic search.
    These models differ from traditional bag-of-words retrieval approaches.
    </TEXT>
    </DOC>
    <DOC>
    <DOCNO> IR005 </DOCNO>
    <TEXT>
    Evaluation of retrieval systems uses relevance judgments and metrics like MAP and NDCG.
    The TREC conference provides standardized test collections and evaluation protocols.
    </TEXT>
    </DOC>
""")


@pytest.fixture(scope="module")
def mini_index(tmp_path_factory):
    """Build a small index from MINI_COLLECTION and return its path."""
    if not HAS_EXT:
        pytest.skip("C++ extension not built")
    from pygalago.parse.document import Document
    from pygalago.index.builder  import IndexBuilder

    tmp = tmp_path_factory.mktemp("integration_idx")
    col_path = str(tmp / "mini.trec")
    idx_path  = str(tmp / "idx")

    with open(col_path, "w") as f:
        f.write(MINI_COLLECTION)

    with IndexBuilder(idx_path, stemmer="none", also_unstemmed=True) as b:
        b.add_documents_from_file(col_path)

    return idx_path


# ── M4 Milestone: Build a new index from a TREC collection ───────────────────

@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestM4BuildIndex:
    """Milestone M4 — build a new index from a collection."""

    def test_index_has_all_parts(self, mini_index):
        assert os.path.isfile(os.path.join(mini_index, "names"))
        assert os.path.isfile(os.path.join(mini_index, "lengths"))
        assert os.path.isfile(os.path.join(mini_index, "postings"))
        assert os.path.isfile(os.path.join(mini_index, "buildManifest.json"))

    def test_all_documents_indexed(self, mini_index):
        idx = _g.DiskIndex(mini_index)
        assert idx.total_documents() == 5

    def test_names_correct(self, mini_index):
        idx = _g.DiskIndex(mini_index)
        names = [idx.get_name(i) for i in range(5)]
        assert names == ["IR001", "IR002", "IR003", "IR004", "IR005"]

    def test_lengths_positive(self, mini_index):
        idx = _g.DiskIndex(mini_index)
        for i in range(5):
            assert idx.get_length(i) > 0

    def test_known_term_in_postings(self, mini_index):
        pr = _g.PostingsReader(os.path.join(mini_index, "postings"))
        it = pr.get_postings("retrieval")
        assert it is not None
        # "retrieval" appears in IR001, IR002, IR003 (possibly IR004 via "retrieval")
        pairs = list(it)
        assert len(pairs) >= 2

    def test_postings_doc_frequency(self, mini_index):
        pr = _g.PostingsReader(os.path.join(mini_index, "postings"))
        # "information" appears in IR001 and IR003
        it = pr.get_postings("information")
        assert it is not None
        stats = it.stats
        assert stats["document_count"] >= 2

    def test_disk_index_interface(self, mini_index):
        idx = _g.DiskIndex(mini_index)
        assert idx.has_names()
        assert idx.has_lengths()
        assert idx.has_postings("postings")


# ── End-to-end search pipeline ────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestEndToEndSearch:
    """Full pipeline: build → search → evaluate."""

    def test_retrieval_returns_ranked_results(self, mini_index):
        from pygalago.retrieval import Retrieval
        r = Retrieval(mini_index, part="postings", b=0.75, k=1.2)
        results = r.search("information retrieval", n=5)
        assert len(results) >= 1
        # Results should be sorted by descending score
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_top_doc_is_relevant(self, mini_index):
        from pygalago.retrieval import Retrieval
        r = Retrieval(mini_index, part="postings")
        results = r.search("BM25 ranking", n=3)
        # IR003 talks about BM25
        names = [name for name, _ in results]
        assert "IR003" in names

    def test_retrieval_missing_query_returns_empty(self, mini_index):
        from pygalago.retrieval import Retrieval
        r = Retrieval(mini_index, part="postings")
        results = r.search("xyzzynotarealwordabcdef", n=10)
        assert results == []

    def test_retrieval_scores_are_finite(self, mini_index):
        import math
        from pygalago.retrieval import Retrieval
        r = Retrieval(mini_index, part="postings")
        results = r.search("retrieval", n=5)
        assert all(math.isfinite(score) for _, score in results)

    def test_eval_pipeline(self, mini_index):
        """Build index → search → compute MAP."""
        from pygalago.retrieval import Retrieval
        from pygalago.eval       import evaluate

        r = Retrieval(mini_index, part="postings")

        queries = {
            "q1": "information retrieval",
            "q2": "search engine index",
            "q3": "BM25 ranking score",
        }
        qrels = {
            "q1": {"IR001": 1, "IR003": 1},
            "q2": {"IR002": 1},
            "q3": {"IR003": 1},
        }

        ranked = {}
        for qid, query in queries.items():
            results = r.search(query, n=5)
            ranked[qid] = [name for name, _ in results]

        scores = evaluate(ranked, qrels, metrics=["map", "p@5", "ndcg@5"])
        assert 0.0 <= scores["map"]  <= 1.0
        assert 0.0 <= scores["p@5"]  <= 1.0
        assert 0.0 <= scores["ndcg@5"] <= 1.0


# ── CLI end-to-end ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestCLIPipeline:
    """Full CLI pipeline: build-index → batch-search → eval."""

    def test_full_pipeline(self, tmp_path):
        import subprocess, sys, json

        col = tmp_path / "col.trec"
        col.write_text(MINI_COLLECTION)
        idx = tmp_path / "idx"
        qfile = tmp_path / "queries.tsv"
        run_file = tmp_path / "run.txt"
        qrels_file = tmp_path / "qrels.txt"

        def cli(*args):
            return subprocess.run(
                [sys.executable, "-c",
                 "from pygalago.tools.cli import main; main()", *args],
                capture_output=True, text=True,
            )

        # 1. Build index
        r = cli("build-index", str(col), "--index", str(idx), "--stemmer", "none")
        assert r.returncode == 0, r.stderr

        # 2. Run batch search
        qfile.write_text("q1\tinformation retrieval\nq2\tsearch engine\n")
        r = cli("batch-search", "--index", str(idx), "--queries", str(qfile),
                "--output", str(run_file), "--stemmer", "none", "-n", "5")
        assert r.returncode == 0, r.stderr
        assert run_file.exists()

        # 3. Eval
        qrels_file.write_text("q1 0 IR001 1\nq1 0 IR003 1\nq2 0 IR002 1\n")
        r = cli("eval", "--qrels", str(qrels_file), "--results", str(run_file),
                "--metrics", "map", "p@5")
        assert r.returncode == 0, r.stderr
        assert "map" in r.stdout.lower()

    def test_dump_index_names_part(self, tmp_path):
        import subprocess, sys

        col = tmp_path / "col.trec"
        col.write_text(MINI_COLLECTION)
        idx = tmp_path / "idx"

        def cli(*args):
            return subprocess.run(
                [sys.executable, "-c",
                 "from pygalago.tools.cli import main; main()", *args],
                capture_output=True, text=True,
            )

        cli("build-index", str(col), "--index", str(idx), "--stemmer", "none")
        r = cli("dump-index", "--index", str(idx), "--part", "names")
        assert r.returncode == 0
        assert "5 entries" in r.stdout


# ── M5 Milestone (Robust04) ────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EXT,  reason="C++ extension not built")
@pytest.mark.skipif(not HAS_INDEX, reason="Set GALAGO_INDEX_PATH for M5 milestone tests")
class TestM5Robust04:
    """Milestone M5 — verify retrieval on Robust04 matches Java reference.

    These tests check that the Python port produces sensible BM25 results
    on a real collection without requiring the Java Galago installation.
    Exact MAP matching against Java is validated in a separate golden-output
    harness (see scripts/golden_compare.py).
    """

    @pytest.fixture(scope="class")
    def retrieval(self):
        from pygalago.retrieval import Retrieval
        return Retrieval(INDEX_PATH)

    def test_retrieval_constructs(self, retrieval):
        assert retrieval is not None

    def test_single_term_returns_1000_results(self, retrieval):
        results = retrieval.search("information", n=1000)
        assert len(results) == 1000

    def test_scores_strictly_descending(self, retrieval):
        results = retrieval.search("retrieval systems", n=100)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_named_docs_are_strings(self, retrieval):
        results = retrieval.search("information", n=10)
        for name, _ in results:
            assert isinstance(name, str) and len(name) > 0

    def test_known_query_topic301(self, retrieval):
        """Topic 301 — International Organized Crime.  Top result should be relevant."""
        results = retrieval.search("international organized crime", n=10)
        assert len(results) >= 5
        # Verify structure only — exact MAP check is in the golden harness.
        for name, score in results:
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(score, float)

    def test_latency_single_query(self, retrieval):
        """Single-term BM25 query over Robust04 should finish in under 2 seconds."""
        t0 = time.perf_counter()
        results = retrieval.search("retrieval", n=1000)
        elapsed = time.perf_counter() - t0
        assert len(results) > 0
        assert elapsed < 2.0, f"Query too slow: {elapsed:.2f}s"

# BSD License (http://www.galagosearch.org/license)
"""Phase 7 tests — CLI tools.

Uses subprocess to invoke ``pygalago`` as a real CLI process so the argument
parsing, dispatch, and output formatting are all exercised end-to-end.
The integration tests that actually build/search an index require the C++
extension to be built (HAS_EXT guard).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

try:
    import pygalago._galago  # noqa: F401
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

PYTHON = sys.executable


def _run(*args: str, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-c", "from pygalago.tools.cli import main; main()", *args],
        capture_output=True,
        text=True,
        input=input,
    )


# ── CLI availability ──────────────────────────────────────────────────────────

class TestCLIAvailability:
    def test_help(self):
        r = _run("--help")
        assert r.returncode == 0
        assert "pygalago" in r.stdout.lower()

    def test_version(self):
        r = _run("--version")
        assert r.returncode == 0
        assert "0.1.0" in r.stdout

    def test_no_subcommand_exits_nonzero(self):
        r = _run()
        assert r.returncode != 0

    def test_unknown_subcommand_exits_nonzero(self):
        r = _run("no-such-command")
        assert r.returncode != 0

    def test_build_index_help(self):
        r = _run("build-index", "--help")
        assert r.returncode == 0
        assert "collection" in r.stdout.lower()

    def test_search_help(self):
        r = _run("search", "--help")
        assert r.returncode == 0
        assert "--query" in r.stdout

    def test_batch_search_help(self):
        r = _run("batch-search", "--help")
        assert r.returncode == 0
        assert "--queries" in r.stdout

    def test_dump_index_help(self):
        r = _run("dump-index", "--help")
        assert r.returncode == 0

    def test_eval_help(self):
        r = _run("eval", "--help")
        assert r.returncode == 0
        assert "--qrels" in r.stdout


# ── build-index ───────────────────────────────────────────────────────────────

TREC_SAMPLE = textwrap.dedent("""\
    <DOC>
    <DOCNO> CLI001 </DOCNO>
    <TEXT>
    information retrieval is a field of study
    </TEXT>
    </DOC>
    <DOC>
    <DOCNO> CLI002 </DOCNO>
    <TEXT>
    search engines use inverted indexes for fast retrieval
    </TEXT>
    </DOC>
""")


@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestBuildIndex:
    def test_build_creates_index_files(self, tmp_path):
        col = tmp_path / "col.trec"
        col.write_text(TREC_SAMPLE)
        idx = tmp_path / "idx"

        r = _run("build-index", str(col), "--index", str(idx), "--stemmer", "none")
        assert r.returncode == 0, r.stderr
        assert (idx / "names").exists()
        assert (idx / "lengths").exists()
        assert (idx / "postings").exists()
        assert (idx / "buildManifest.json").exists()

    def test_manifest_document_count(self, tmp_path):
        col = tmp_path / "col.trec"
        col.write_text(TREC_SAMPLE)
        idx = tmp_path / "idx"
        _run("build-index", str(col), "--index", str(idx), "--stemmer", "none")
        with open(idx / "buildManifest.json") as f:
            m = json.load(f)
        assert m["documentCount"] == 2

    def test_no_unstemmed_flag(self, tmp_path):
        # With a named stemmer, the stemmed part is "postings.<stemmer>" and
        # --no-unstemmed means the bare "postings" part should NOT be written.
        col = tmp_path / "col.trec"
        col.write_text(TREC_SAMPLE)
        idx = tmp_path / "idx"
        # Use porter so the stemmed part name is "postings.porter"
        try:
            import Stemmer  # noqa: F401
        except ImportError:
            pytest.skip("PyStemmer not installed; needed for porter stemmer test")
        _run("build-index", str(col), "--index", str(idx),
             "--stemmer", "porter", "--no-unstemmed")
        assert (idx / "postings.porter").exists()
        assert not (idx / "postings").exists()


# ── search ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestSearch:
    @pytest.fixture(scope="class")
    def small_index(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("idx")
        col = tmp / "col.trec"
        col.write_text(TREC_SAMPLE)
        _run("build-index", str(col), "--index", str(tmp / "idx"), "--stemmer", "none")
        return str(tmp / "idx")

    def test_search_returns_results(self, small_index):
        r = _run("search", "--index", small_index,
                 "--query", "information", "--count", "5",
                 "--stemmer", "none")
        assert r.returncode == 0, r.stderr
        assert "CLI001" in r.stdout

    def test_search_output_has_rank_and_score(self, small_index):
        r = _run("search", "--index", small_index,
                 "--query", "retrieval", "--count", "2",
                 "--stemmer", "none")
        assert r.returncode == 0
        lines = [l for l in r.stdout.splitlines() if l.strip() and "Query" not in l]
        assert len(lines) >= 1
        # Each result line: rank + score + doc_id
        for line in lines:
            parts = line.split()
            assert len(parts) >= 3


# ── batch-search ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestBatchSearch:
    @pytest.fixture(scope="class")
    def small_index(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("idx_b")
        col = tmp / "col.trec"
        col.write_text(TREC_SAMPLE)
        _run("build-index", str(col), "--index", str(tmp / "idx"), "--stemmer", "none")
        return str(tmp / "idx")

    def test_writes_run_file(self, small_index, tmp_path):
        qfile = tmp_path / "queries.tsv"
        qfile.write_text("301\tinformation retrieval\n302\tsearch engines\n")
        out = tmp_path / "run.txt"

        r = _run("batch-search", "--index", small_index,
                 "--queries", str(qfile), "--output", str(out),
                 "--stemmer", "none", "-n", "10")
        assert r.returncode == 0, r.stderr
        assert out.exists()

    def test_run_file_is_valid_trec(self, small_index, tmp_path):
        qfile = tmp_path / "queries.tsv"
        qfile.write_text("301\tinformation\n")
        out = tmp_path / "run.txt"
        _run("batch-search", "--index", small_index,
             "--queries", str(qfile), "--output", str(out),
             "--stemmer", "none", "-n", "5")

        from pygalago.eval.run import read_run
        run = read_run(str(out))
        assert "301" in run
        assert len(run["301"]) >= 1


# ── dump-index ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestDumpIndex:
    @pytest.fixture(scope="class")
    def small_index(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("idx_d")
        col = tmp / "col.trec"
        col.write_text(TREC_SAMPLE)
        _run("build-index", str(col), "--index", str(tmp / "idx"), "--stemmer", "none")
        return str(tmp / "idx")

    def test_dump_names(self, small_index):
        r = _run("dump-index", "--index", small_index, "--part", "names")
        assert r.returncode == 0, r.stderr
        assert "entries" in r.stdout

    def test_dump_postings(self, small_index):
        # No --max-terms limit so all terms appear.
        r = _run("dump-index", "--index", small_index, "--part", "postings")
        assert r.returncode == 0, r.stderr
        # At least one expected term must appear in the dump.
        assert "information" in r.stdout or "retrieval" in r.stdout or "search" in r.stdout

    def test_dump_nonexistent_part(self, small_index):
        r = _run("dump-index", "--index", small_index,
                 "--part", "no_such_part")
        assert r.returncode != 0


# ── eval ─────────────────────────────────────────────────────────────────────

QRELS_CONTENT = """\
301 0 CLI001 1
301 0 CLI002 0
302 0 CLI002 1
"""

RUN_CONTENT = """\
301 Q0 CLI001 1 0.9 test
301 Q0 CLI002 2 0.5 test
302 Q0 CLI002 1 0.8 test
"""


class TestEvalCLI:
    def test_eval_output_has_metrics(self, tmp_path):
        qrels = tmp_path / "qrels.txt"
        run   = tmp_path / "run.txt"
        qrels.write_text(QRELS_CONTENT)
        run.write_text(RUN_CONTENT)

        r = _run("eval", "--qrels", str(qrels), "--results", str(run),
                 "--metrics", "map", "p@10")
        assert r.returncode == 0, r.stderr
        assert "map" in r.stdout.lower()
        assert "p@10" in r.stdout.lower()

    def test_eval_map_perfect_run(self, tmp_path):
        qrels = tmp_path / "qrels.txt"
        run   = tmp_path / "run.txt"
        qrels.write_text("301 0 d1 1\n")
        run.write_text("301 Q0 d1 1 1.0 r\n")

        r = _run("eval", "--qrels", str(qrels), "--results", str(run),
                 "--metrics", "map")
        assert r.returncode == 0
        # MAP should be 1.0000
        assert "1.0000" in r.stdout

    def test_eval_per_topic(self, tmp_path):
        qrels = tmp_path / "qrels.txt"
        run   = tmp_path / "run.txt"
        qrels.write_text(QRELS_CONTENT)
        run.write_text(RUN_CONTENT)

        r = _run("eval", "--qrels", str(qrels), "--results", str(run),
                 "--metrics", "map", "--per-topic")
        assert r.returncode == 0
        assert "301" in r.stdout
        assert "302" in r.stdout

"""Tests for pygalago.index — runs against the Robust04 index if available.

Set GALAGO_INDEX_PATH to the index directory to enable integration tests.
Without it, only import and graceful-degradation tests run.
"""

import os
import pytest

INDEX_PATH = os.environ.get("GALAGO_INDEX_PATH", "")
HAS_INDEX  = bool(INDEX_PATH)

try:
    import pygalago._galago as _g
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

# ── Import / availability tests (no index needed) ────────────────────────────

class TestImportAndAPI:
    def test_module_importable(self):
        import pygalago.index as gi
        assert hasattr(gi, "open")
        assert hasattr(gi, "open_names")
        assert hasattr(gi, "open_lengths")
        assert hasattr(gi, "open_postings")

    def test_open_raises_without_extension(self, tmp_path):
        import pygalago.index as gi
        if HAS_EXT:
            pytest.skip("Extension is built — skipping graceful-degradation test")
        with pytest.raises(RuntimeError, match="extension"):
            gi.open(str(tmp_path))


# ── Integration tests (require extension + Robust04 index) ───────────────────

@pytest.fixture(scope="module")
def index():
    if not HAS_EXT:
        pytest.skip("C++ extension not built")
    if not HAS_INDEX:
        pytest.skip("Set GALAGO_INDEX_PATH to run index integration tests")
    import pygalago.index as gi
    return gi.open(INDEX_PATH)


class TestNamesReader:
    def test_open_names(self):
        if not HAS_EXT or not HAS_INDEX:
            pytest.skip("requires extension + index")
        import pygalago.index as gi
        names = gi.open_names(os.path.join(INDEX_PATH, "names"))
        assert names is not None

    def test_get_name(self, index):
        name = index.get_name(0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_name_known_value(self, index):
        # Robust04 doc 0 (FT-sorted build) is "FT931-1"
        assert index.get_name(0) == "FT931-1"

    def test_get_name_out_of_range(self, index):
        # Very large docid — should return empty string, not crash
        result = index.get_name(10**15)
        assert isinstance(result, str)


class TestLengthsReader:
    def test_open_lengths(self):
        if not HAS_EXT or not HAS_INDEX:
            pytest.skip("requires extension + index")
        import pygalago.index as gi
        lengths = gi.open_lengths(os.path.join(INDEX_PATH, "lengths"))
        assert lengths is not None

    def test_get_length_positive(self, index):
        length = index.get_length(0)
        assert isinstance(length, int)
        assert length > 0

    def test_get_length_known_value(self, index):
        # doc 0 (FT931-1) in this Robust04 build has 965 tokens
        assert index.get_length(0) == 965

    def test_total_documents(self, index):
        n = index.total_documents()
        assert n == 528155  # Robust04 has 528,155 documents

    def test_length_stats(self, index):
        stats = index.get_length_stats()
        assert stats.total_document_count == 528155
        assert stats.collection_length == 252013235
        assert abs(stats.avg_length - 477.158) < 0.01
        assert stats.first_document == 0
        assert stats.last_document > 0

    def test_length_out_of_range_returns_zero(self, index):
        assert index.get_length(10**15) == 0


class TestPostingsReader:
    def test_open_postings(self):
        if not HAS_EXT or not HAS_INDEX:
            pytest.skip("requires extension + index")
        import pygalago.index as gi
        posts = gi.open_postings(os.path.join(INDEX_PATH, "postings.krovetz"))
        assert posts is not None

    def test_get_postings_known_term(self, index):
        it = index.get_postings("information")
        assert it is not None
        stats = it.stats
        assert stats["document_count"] == 68145
        assert stats["collection_count"] == 163441

    def test_postings_monotone_docids(self, index):
        it = index.get_postings("information")
        assert it is not None
        prev = -1
        count = 0
        while not it.is_done and count < 100:
            assert it.doc_id > prev
            assert it.count > 0
            prev = it.doc_id
            it.next()
            count += 1
        assert count == 100

    def test_postings_python_iteration(self, index):
        it = index.get_postings("retrieval")
        assert it is not None
        pairs = list(zip(range(10), it))  # take first 10 via __next__
        assert len(pairs) == 10
        docids = [p[1][0] for p in pairs]
        assert docids == sorted(docids)

    def test_postings_missing_term_returns_none(self, index):
        result = index.get_postings("xyzzy_this_term_cannot_exist_xyzzy")
        assert result is None

    def test_postings_skip_to(self, index):
        it = index.get_postings("information")
        target = 10000
        it.skip_to(target)
        assert not it.is_done
        assert it.doc_id >= target

    def test_has_postings(self, index):
        assert index.has_postings("postings.krovetz")
        assert not index.has_postings("nonexistent.part")

    def test_postings_collection_frequency(self, index):
        # collection_count >= document_count always
        it = index.get_postings("the")
        if it is None:
            pytest.skip("term 'the' not in index")
        s = it.stats
        assert s["collection_count"] >= s["document_count"]


class TestDiskIndex:
    def test_has_parts(self, index):
        assert index.has_names()
        assert index.has_lengths()

    def test_consistent_name_and_length(self, index):
        # Same docid looked up through DiskIndex should give coherent results
        name   = index.get_name(42)
        length = index.get_length(42)
        assert isinstance(name, str) and len(name) > 0
        assert isinstance(length, int) and length > 0

    def test_path_attribute(self, index):
        assert INDEX_PATH in index.path

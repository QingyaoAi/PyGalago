# BSD License (http://www.galagosearch.org/license)
"""Phase 5 tests — document parsing, tokenisation, stemming, and index building.

The parser and tokeniser tests run without any C++ extension.
The IndexBuilder round-trip test requires the extension to be built.
"""

from __future__ import annotations

import io
import json
import os
import textwrap

import pytest

try:
    import pygalago._galago as _g
    HAS_EXT = True
except ImportError:
    HAS_EXT = False


# ── Document model ────────────────────────────────────────────────────────────

class TestDocument:
    def test_defaults(self):
        from pygalago.parse.document import Document
        doc = Document()
        assert doc.name == ""
        assert doc.text == ""
        assert doc.terms == []
        assert doc.identifier == -1

    def test_from_text(self):
        from pygalago.parse.document import Document
        doc = Document.from_text("doc1", "hello world")
        assert doc.name == "doc1"
        assert doc.text == "hello world"


# ── Tokeniser ─────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_simple(self):
        from pygalago.parse.tokenizer import tokenize_string
        assert tokenize_string("Hello World") == ["hello", "world"]

    def test_strips_html(self):
        from pygalago.parse.tokenizer import tokenize_string
        toks = tokenize_string("<b>bold</b> text")
        assert toks == ["bold", "text"]

    def test_hyphenated(self):
        from pygalago.parse.tokenizer import tokenize_string
        toks = tokenize_string("state-of-the-art")
        assert "state-of-the-art" in toks or len(toks) > 1

    def test_tokenize_doc_inplace(self):
        from pygalago.parse.document import Document
        from pygalago.parse.tokenizer import tokenize
        doc = Document.from_text("d1", "Information Retrieval")
        tokenize(doc)
        assert doc.terms == ["information", "retrieval"]

    def test_empty_string(self):
        from pygalago.parse.tokenizer import tokenize_string
        assert tokenize_string("") == []

    def test_numbers_kept(self):
        from pygalago.parse.tokenizer import tokenize_string
        toks = tokenize_string("year 2024 model")
        assert "2024" in toks


# ── Stemmer ───────────────────────────────────────────────────────────────────

class TestStemmer:
    def test_none_stemmer_is_identity(self):
        from pygalago.parse.stemmer import get_stemmer
        stem = get_stemmer("none")
        assert stem("running") == "running"

    def test_porter_if_available(self):
        try:
            import Stemmer  # noqa: F401
        except ImportError:
            pytest.skip("PyStemmer not installed")
        from pygalago.parse.stemmer import get_stemmer
        stem = get_stemmer("porter")
        assert stem("information") != "information"  # should stem

    def test_stem_terms(self):
        from pygalago.parse.stemmer import stem_terms
        result = stem_terms(["running", "dogs"], "none")
        assert result == ["running", "dogs"]

    def test_unknown_name_returns_identity(self):
        from pygalago.parse.stemmer import get_stemmer
        stem = get_stemmer("unknown_stemmer_xyz")
        assert stem("hello") == "hello"


# ── TREC text parser ──────────────────────────────────────────────────────────

TREC_SAMPLE = textwrap.dedent("""\
    <DOC>
    <DOCNO> DOC001 </DOCNO>
    <TEXT>
    Information retrieval is a field of study.
    </TEXT>
    </DOC>
    <DOC>
    <DOCNO> DOC002 </DOCNO>
    <HEADLINE>
    Second Document Headline
    </HEADLINE>
    <TEXT>
    Another document about search engines.
    </TEXT>
    </DOC>
""")


class TestTrecTextParser:
    def test_yields_two_docs(self):
        from pygalago.parse.parsers.trec_text import parse_stream
        docs = list(parse_stream(io.StringIO(TREC_SAMPLE)))
        assert len(docs) == 2

    def test_doc_names(self):
        from pygalago.parse.parsers.trec_text import parse_stream
        docs = list(parse_stream(io.StringIO(TREC_SAMPLE)))
        assert docs[0].name == "DOC001"
        assert docs[1].name == "DOC002"

    def test_doc_text_nonempty(self):
        from pygalago.parse.parsers.trec_text import parse_stream
        docs = list(parse_stream(io.StringIO(TREC_SAMPLE)))
        assert "information" in docs[0].text.lower()
        assert "search" in docs[1].text.lower()

    def test_headline_captured(self):
        from pygalago.parse.parsers.trec_text import parse_stream
        docs = list(parse_stream(io.StringIO(TREC_SAMPLE)))
        assert "Headline" in docs[1].text or "headline" in docs[1].text.lower()

    def test_empty_stream(self):
        from pygalago.parse.parsers.trec_text import parse_stream
        docs = list(parse_stream(io.StringIO("")))
        assert docs == []

    def test_doc_without_docno_skipped(self):
        bad = "<DOC>\n<TEXT>no docno here</TEXT>\n</DOC>\n"
        from pygalago.parse.parsers.trec_text import parse_stream
        docs = list(parse_stream(io.StringIO(bad)))
        assert docs == []


# ── TREC web parser ───────────────────────────────────────────────────────────

TREC_WEB_SAMPLE = textwrap.dedent("""\
    <DOC>
    <DOCNO>WTX001-B01-1</DOCNO>
    <DOCHDR>
    http://www.example.com/
    </DOCHDR>
    <html><body>Hello web world</body></html>
    </DOC>
""")


class TestTrecWebParser:
    def test_yields_one_doc(self):
        from pygalago.parse.parsers.trec_web import parse_stream
        docs = list(parse_stream(io.StringIO(TREC_WEB_SAMPLE)))
        assert len(docs) == 1

    def test_docno_and_url(self):
        from pygalago.parse.parsers.trec_web import parse_stream
        docs = list(parse_stream(io.StringIO(TREC_WEB_SAMPLE)))
        assert docs[0].name == "WTX001-B01-1"
        assert "example.com" in docs[0].metadata.get("url", "")

    def test_body_in_text(self):
        from pygalago.parse.parsers.trec_web import parse_stream
        docs = list(parse_stream(io.StringIO(TREC_WEB_SAMPLE)))
        assert "Hello" in docs[0].text


# ── JSON parser ───────────────────────────────────────────────────────────────

class TestJsonParser:
    def test_jsonlines(self):
        from pygalago.parse.parsers.json_parser import parse_stream
        raw = '{"id":"d1","text":"hello world"}\n{"id":"d2","text":"foo bar"}\n'
        docs = list(parse_stream(io.StringIO(raw)))
        assert len(docs) == 2
        assert docs[0].name == "d1"
        assert "hello" in docs[0].text

    def test_json_array(self):
        from pygalago.parse.parsers.json_parser import parse_stream
        raw = json.dumps([
            {"docno": "x1", "body": "the quick brown fox"},
            {"docid": "x2", "contents": "jumps over"},
        ])
        docs = list(parse_stream(io.StringIO(raw)))
        assert len(docs) == 2
        assert docs[0].name == "x1"

    def test_missing_id_skipped(self):
        from pygalago.parse.parsers.json_parser import parse_stream
        raw = '{"text":"no id field here"}\n'
        docs = list(parse_stream(io.StringIO(raw)))
        assert docs == []


# ── WARC parser ───────────────────────────────────────────────────────────────

def _make_warc_record(trec_id: str, uri: str, body: str) -> bytes:
    http_resp = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{body}"
    content   = http_resp.encode("utf-8")
    header = (
        f"WARC/1.0\r\n"
        f"WARC-Type: response\r\n"
        f"WARC-TREC-ID: {trec_id}\r\n"
        f"WARC-Target-URI: {uri}\r\n"
        f"Content-Length: {len(content)}\r\n"
        f"\r\n"
    ).encode("ascii")
    return header + content + b"\r\n\r\n"


class TestWarcParser:
    def test_yields_response_records(self):
        from pygalago.parse.parsers.warc import parse_stream
        raw = _make_warc_record("clueweb09-en0000-01-00001",
                                "http://example.com/", "Hello WARC world")
        docs = list(parse_stream(io.BytesIO(raw)))
        assert len(docs) == 1
        assert docs[0].name == "clueweb09-en0000-01-00001"
        assert "Hello" in docs[0].text

    def test_non_response_records_skipped(self):
        from pygalago.parse.parsers.warc import parse_stream
        content = b"Some warcinfo content"
        header = (
            f"WARC/1.0\r\nWARC-Type: warcinfo\r\n"
            f"Content-Length: {len(content)}\r\n\r\n"
        ).encode("ascii")
        raw = header + content + b"\r\n\r\n"
        docs = list(parse_stream(io.BytesIO(raw)))
        assert docs == []

    def test_url_in_metadata(self):
        from pygalago.parse.parsers.warc import parse_stream
        raw = _make_warc_record("doc1", "http://test.org/page", "body text")
        docs = list(parse_stream(io.BytesIO(raw)))
        assert docs[0].metadata.get("url") == "http://test.org/page"

    def test_multiple_records(self):
        from pygalago.parse.parsers.warc import parse_stream
        records = b"".join(
            _make_warc_record(f"doc{i}", f"http://x.com/{i}", f"doc {i} body")
            for i in range(3)
        )
        docs = list(parse_stream(io.BytesIO(records)))
        assert len(docs) == 3
        assert [d.name for d in docs] == ["doc0", "doc1", "doc2"]


# ── open_collection dispatcher ────────────────────────────────────────────────

class TestOpenCollection:
    def test_trec_file(self, tmp_path):
        p = tmp_path / "col.trec"
        p.write_text(TREC_SAMPLE)
        from pygalago.parse.parsers import open_collection
        docs = list(open_collection(str(p)))
        assert len(docs) == 2

    def test_json_file(self, tmp_path):
        p = tmp_path / "col.json"
        p.write_text('[{"id":"j1","text":"json doc"}]')
        from pygalago.parse.parsers import open_collection
        docs = list(open_collection(str(p)))
        assert len(docs) == 1 and docs[0].name == "j1"

    def test_warc_file(self, tmp_path):
        p = tmp_path / "col.warc"
        p.write_bytes(_make_warc_record("w1", "http://a.com/", "warc body"))
        from pygalago.parse.parsers import open_collection
        docs = list(open_collection(str(p)))
        assert len(docs) == 1 and docs[0].name == "w1"


# ── IndexBuilder round-trip ───────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EXT, reason="C++ extension not built")
class TestIndexBuilderRoundTrip:
    """Build a small index from scratch, then read it back with the readers."""

    DOCS = [
        ("DOC001", "information retrieval is a field of study"),
        ("DOC002", "search engines use inverted indexes"),
        ("DOC003", "information about search and retrieval systems"),
    ]

    @pytest.fixture(scope="class")
    def built_index(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("idx")
        from pygalago.parse.document import Document
        from pygalago.index.builder import IndexBuilder

        with IndexBuilder(str(tmp), stemmer="none", also_unstemmed=True) as builder:
            for name, text in self.DOCS:
                builder.add_document(Document.from_text(name, text))

        return str(tmp)

    def test_manifest_written(self, built_index):
        manifest = os.path.join(built_index, "buildManifest.json")
        assert os.path.isfile(manifest)
        with open(manifest) as f:
            m = json.load(f)
        assert m["documentCount"] == 3

    def test_names_file_exists(self, built_index):
        assert os.path.isfile(os.path.join(built_index, "names"))

    def test_lengths_file_exists(self, built_index):
        assert os.path.isfile(os.path.join(built_index, "lengths"))

    def test_postings_file_exists(self, built_index):
        assert os.path.isfile(os.path.join(built_index, "postings"))

    def test_names_round_trip(self, built_index):
        nr = _g.NamesReader(os.path.join(built_index, "names"))
        assert nr.get_name(0) == "DOC001"
        assert nr.get_name(1) == "DOC002"
        assert nr.get_name(2) == "DOC003"

    def test_lengths_round_trip(self, built_index):
        lr = _g.LengthsReader(os.path.join(built_index, "lengths"))
        # "information retrieval is a field of study" → 7 tokens
        assert lr.get_length(0) == 7
        # "search engines use inverted indexes" → 5 tokens
        assert lr.get_length(1) == 5

    def test_lengths_stats(self, built_index):
        lr = _g.LengthsReader(os.path.join(built_index, "lengths"))
        stats = lr.get_stats("document")
        assert stats.total_document_count == 3
        assert stats.collection_length == 7 + 5 + 6  # DOC001=7, DOC002=5, DOC003=6
        assert stats.first_document == 0
        assert stats.last_document == 2

    def test_postings_known_term(self, built_index):
        pr = _g.PostingsReader(os.path.join(built_index, "postings"))
        it = pr.get_postings("information")
        assert it is not None
        s = it.stats
        assert s["document_count"] == 2   # DOC001 and DOC003
        assert s["collection_count"] == 2

    def test_postings_monotone_docids(self, built_index):
        pr = _g.PostingsReader(os.path.join(built_index, "postings"))
        it = pr.get_postings("search")
        assert it is not None
        docids = [d for d, _ in it]
        assert docids == sorted(docids)
        assert len(docids) == 2  # DOC002 and DOC003

    def test_missing_term_returns_none(self, built_index):
        pr = _g.PostingsReader(os.path.join(built_index, "postings"))
        assert pr.get_postings("xyzzy_not_in_index") is None

    def test_disk_index_open(self, built_index):
        idx = _g.DiskIndex(built_index)
        assert idx.get_name(0) == "DOC001"
        assert idx.get_length(0) == 7
        assert idx.total_documents() == 3

    def test_add_documents_from_file(self, tmp_path):
        trec_file = tmp_path / "mini.trec"
        trec_file.write_text(TREC_SAMPLE)
        out_dir = tmp_path / "idx2"

        from pygalago.index.builder import IndexBuilder
        with IndexBuilder(str(out_dir), stemmer="none", also_unstemmed=True) as b:
            count = b.add_documents_from_file(str(trec_file))

        assert count == 2
        nr = _g.NamesReader(str(out_dir / "names"))
        assert nr.get_name(0) == "DOC001"
        assert nr.get_name(1) == "DOC002"

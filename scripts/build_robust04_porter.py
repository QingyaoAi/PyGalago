#!/usr/bin/env python3
# BSD License (http://www.galagosearch.org/license)
"""Build a Galago-format Porter2 index with positional postings from raw TREC corpus.

Uses a streaming external-sort approach to keep peak RAM well under 2 GB:
  Pass 1  — parse every document, emit (term, docid, positions) as text lines
             to a temp file (O(1) RAM, one doc in memory at a time).
  Sort    — sort the temp file by (term lexicographic, docid numeric) with the
             OS-level `sort` command.
  Pass 2  — stream through the sorted file, group by term, encode each term's
             positional posting list and write it directly to the B-tree
             using the incremental DiskBTreeWriter.

Also writes `names` and `lengths` index parts.

Output index layout:
  <output_dir>/
    names
    lengths
    postings.porter   ← positional Porter2 postings
    buildManifest.json

Usage
-----
    python scripts/build_robust04_porter.py [--output /path/to/index]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygalago._galago as g
from pygalago.parse.tokenizer import tokenize_string
from pygalago.parse.stemmer   import get_stemmer
from pygalago.parse.parsers.trec_text import parse_file

# ── Configuration ─────────────────────────────────────────────────────────────

CORPUS_FILES = [
    "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-trec/FBIS.trec",
    "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-trec/FR9.trec",
    "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-trec/FT.trec",
    "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-trec/LA.trec",
]

DEFAULT_OUTPUT = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-porter.index"

# Manifest constants — outer JSON braces must be doubled for .format()
POSTINGS_MANIFEST = (
    '{{"writerClass":"PositionIndexWriter","readerClass":"PositionIndexReader",'
    '"defaultOperator":"counts",'
    '"statistics/collectionLength":{coll_len},'
    '"statistics/vocabCount":{vocab_count},'
    '"statistics/highestDocumentCount":{max_df},'
    '"statistics/highestFrequency":{max_cf},'
    '"documentCount":{doc_count}}}'
)


# ── Pass 1: parse corpus → temp file ─────────────────────────────────────────

def pass1_write_triples(tmp_path: str, stem_fn) -> tuple[List[str], List[int], int]:
    """Stream all documents, write (term, docid, pos_str) lines to tmp_path.

    Returns (names, lengths, total_tokens).
    """
    names: List[str]  = []
    lengths: List[int] = []
    total_tokens = 0
    docid = 0
    t0 = time.perf_counter()

    with open(tmp_path, "w", encoding="utf-8", buffering=1 << 22) as out:
        for corpus_file in CORPUS_FILES:
            print(f"  [{docid:>6,}] Parsing {os.path.basename(corpus_file)} …",
                  flush=True)
            for doc in parse_file(corpus_file):
                tokens = tokenize_string(doc.text)
                lengths.append(len(tokens))
                total_tokens += len(tokens)
                names.append(doc.name)

                # Collect positions per stem for this doc
                term_positions: Dict[str, List[int]] = defaultdict(list)
                for pos, tok in enumerate(tokens):
                    s = stem_fn(tok)
                    if s:
                        term_positions[s].append(pos)

                for term, positions in term_positions.items():
                    # One line per (docid, term): term \t docid \t p1,p2,...
                    out.write(f"{term}\t{docid}\t{','.join(map(str, positions))}\n")

                docid += 1
                if docid % 50_000 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = docid / elapsed
                    print(f"  [{docid:>6,}] {elapsed:.0f}s  ({rate:.0f} docs/s)",
                          flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Pass 1 done: {docid:,} docs, {total_tokens:,} tokens in {elapsed:.1f}s")
    return names, lengths, total_tokens


# ── Sort ───────────────────────────────────────────────────────────────────────

def sort_triples(tmp_path: str, sorted_path: str) -> None:
    """Sort temp file by (term, docid) using OS sort."""
    print("Sorting …", flush=True)
    t0 = time.perf_counter()
    file_mb = os.path.getsize(tmp_path) / 1e6
    print(f"  Temp file size: {file_mb:.0f} MB")

    # Use the system sort command: -k1,1 (term) then -k2,2n (numeric docid)
    # LC_ALL=C for byte-order sort (faster, consistent)
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    subprocess.run(
        ["sort", "-k1,1", "-k2,2n", f"-t\t",
         "-S", "4G",       # allow 4 GB sort buffer
         tmp_path, "-o", sorted_path],
        check=True,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    print(f"Sort done in {elapsed:.1f}s")


# ── Pass 2: stream sorted triples → write index ───────────────────────────────

def pass2_write_postings(sorted_path: str, output_path: str,
                         total_docs: int, total_tokens: int) -> None:
    """Read sorted (term, docid, positions) and write postings.porter B-tree."""
    postings_path = os.path.join(output_path, "postings.porter")
    print(f"Writing {postings_path} …", flush=True)
    t0 = time.perf_counter()

    manifest_template = POSTINGS_MANIFEST

    vocab_count = 0
    max_df = 0
    max_cf = 0

    # We need to compute manifest stats AFTER writing; use a two-step approach:
    # collect stats during write, then we can't update the manifest in-place.
    # Instead, build a small stats dict and write after close.
    # The BTreeWriter takes the manifest at close() time, so we:
    #   1. Write all entries (accumulating stats)
    #   2. Close with the final manifest.

    # Hold one term's postings in memory at a time (max ~528K entries for "the")
    current_term: str | None = None
    current_postings: List[tuple] = []   # [(docid, [pos, ...])]

    writer = g.BTreeWriter(postings_path)

    def flush_term(term: str, postings: list) -> tuple:
        """Encode and write one term's posting list; return (df, cf)."""
        encoded = g.encode_positional_postings(postings)
        writer.add(term, encoded)
        df = len(postings)
        cf = sum(len(ps) for _, ps in postings)
        return df, cf

    terms_written = 0
    with open(sorted_path, encoding="utf-8", buffering=1 << 22) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            term, docid_str, pos_str = parts
            docid    = int(docid_str)
            positions = [int(p) for p in pos_str.split(",") if p]

            if term != current_term:
                if current_term is not None:
                    df, cf = flush_term(current_term, current_postings)
                    vocab_count += 1
                    if df > max_df: max_df = df
                    if cf > max_cf: max_cf = cf
                    terms_written += 1
                    if terms_written % 20_000 == 0:
                        elapsed = time.perf_counter() - t0
                        print(f"  {terms_written:,} terms  ({elapsed:.0f}s)",
                              flush=True)
                current_term = term
                current_postings = []

            current_postings.append((docid, positions))

    # Flush last term
    if current_term is not None:
        df, cf = flush_term(current_term, current_postings)
        vocab_count += 1
        if df > max_df: max_df = df
        if cf > max_cf: max_cf = cf

    manifest = manifest_template.format(
        coll_len=total_tokens,
        vocab_count=vocab_count,
        max_df=max_df,
        max_cf=max_cf,
        doc_count=total_docs,
    )
    writer.close(manifest)

    elapsed = time.perf_counter() - t0
    print(f"Pass 2 done: {vocab_count:,} terms in {elapsed:.1f}s")


# ── Write names + lengths ─────────────────────────────────────────────────────

def write_meta(output_path: str, names: List[str], lengths: List[int],
               total_tokens: int) -> None:
    print(f"Writing names ({len(names):,} docs) …")
    g.write_names(os.path.join(output_path, "names"), names)

    print(f"Writing lengths …")
    g.write_lengths(os.path.join(output_path, "lengths"),
                    [int(l) for l in lengths])


def write_manifest(output_path: str, n_docs: int, total_tokens: int) -> None:
    manifest = {
        "indexPath":        output_path,
        "documentCount":    n_docs,
        "collectionLength": total_tokens,
        "stemmer":          "porter",
        "withPositions":    True,
        "buildTime":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder":          "PyGalago build_robust04_porter.py",
    }
    with open(os.path.join(output_path, "buildManifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def build(output_path: str, keep_temp: bool = True) -> None:
    """Build the index, using persistent temp files so partial runs can resume.

    Intermediate files are stored at <output_path>/.build_tmp/ and reused if
    they already exist, allowing the sort step to be skipped on re-runs.
    Pass --no-keep-temp to delete them on successful completion.
    """
    os.makedirs(output_path, exist_ok=True)
    stem_fn = get_stemmer("porter")

    tmpdir      = os.path.join(output_path, ".build_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    tmp_path    = os.path.join(tmpdir, "triples.tsv")
    sorted_path = os.path.join(tmpdir, "triples_sorted.tsv")
    meta_path   = os.path.join(tmpdir, "meta.json")

    total_t0 = time.perf_counter()

    # ── Pass 1 (skip if sorted file exists) ──────────────────────────────────
    if os.path.exists(sorted_path) and os.path.exists(meta_path):
        print("=== Resuming: sorted file found, skipping Pass 1 + sort ===")
        with open(meta_path) as f:
            meta = json.load(f)
        names         = meta["names"]
        lengths       = meta["lengths"]
        total_tokens  = meta["total_tokens"]
    else:
        print("=== Pass 1: parse corpus → temp file ===")
        names, lengths, total_tokens = pass1_write_triples(tmp_path, stem_fn)

        # Persist meta so we can resume after a pass-2 failure
        with open(meta_path, "w") as f:
            json.dump({"names": names, "lengths": lengths,
                       "total_tokens": total_tokens}, f)

        print("\n=== Sorting ===")
        sort_triples(tmp_path, sorted_path)

        # Remove the unsorted file to free space
        os.remove(tmp_path)

    # ── Pass 2 ────────────────────────────────────────────────────────────────
    print("\n=== Pass 2: write postings.porter ===")
    pass2_write_postings(sorted_path, output_path, len(names), total_tokens)

    print("\n=== Writing names + lengths ===")
    write_meta(output_path, names, lengths, total_tokens)
    write_manifest(output_path, len(names), total_tokens)

    if not keep_temp:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Temp files removed.")

    elapsed = time.perf_counter() - total_t0
    print(f"\n✓ Index built at {output_path!r}")
    print(f"  {len(names):,} documents, {total_tokens:,} tokens in {elapsed/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Robust04 Porter2 positional index")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output index directory (default: {DEFAULT_OUTPUT})")
    args = ap.parse_args()
    build(args.output)

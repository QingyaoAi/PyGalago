#!/usr/bin/env python3
"""Extract the Robust04 corpus from Galago corpus files to TREC format.

Galago corpus format (per document):
  Full stream  : \x82SNAPPY\x00 + version(4) + min_version(4) + compressed_len(4) + snappy_data
  Continuation : compressed_len(4) + snappy_data   (no magic; for docs > 32768 decompressed bytes)

Each full stream decompresses to:
  16-byte block header + [4-byte name_len][name][8-byte content_len][content...]

Large docs span multiple continuation blocks; content bytes are simply concatenated.
"""
from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

import snappy

MAGIC      = b'\x82SNAPPY\x00'
HEADER_LEN = 20   # magic(8) + version(4) + min_version(4) + compressed_len(4)
BLOCK_HDR  = 16   # metadata prefix inside each decompressed block


def iter_documents(corpus_file: str):
    """Yield (doc_name, doc_content_bytes) for every document in one corpus split."""
    with open(corpus_file, "rb") as f:
        data = f.read()

    offset = 0
    n = len(data)
    current_name: str | None = None
    content_chunks: list[bytes] = []
    remaining: int = 0          # bytes still needed for current doc

    while offset < n:
        if data[offset:offset+8] == MAGIC:
            # ── Full stream: start of a new document ──────────────────────────
            if current_name is not None:
                yield current_name, b"".join(content_chunks)

            compressed_len = struct.unpack(">I", data[offset+16:offset+20])[0]
            block_bytes = snappy.decompress(
                data[offset+20 : offset+20+compressed_len]
            )
            offset += HEADER_LEN + compressed_len

            # Parse document record from decompressed block
            pos = BLOCK_HDR
            name_len = struct.unpack(">I", block_bytes[pos:pos+4])[0]
            current_name = block_bytes[pos+4 : pos+4+name_len].decode()
            # content_len stored as int64 (8 bytes)
            content_len = struct.unpack(
                ">Q", block_bytes[pos+4+name_len : pos+4+name_len+8]
            )[0]
            chunk = block_bytes[pos+4+name_len+8:]
            content_chunks = [chunk]
            remaining = content_len - len(chunk)

        elif remaining > 0:
            # ── Continuation block (large doc spanning multiple blocks) ───────
            compressed_len = struct.unpack(">I", data[offset:offset+4])[0]
            chunk = snappy.decompress(
                data[offset+4 : offset+4+compressed_len]
            )
            offset += 4 + compressed_len
            content_chunks.append(chunk)
            remaining -= len(chunk)

        else:
            # Trailing padding bytes at EOF — stop
            break

    if current_name is not None:
        yield current_name, b"".join(content_chunks)


def extract_corpus(corpus_dir: str, output_dir: str) -> None:
    corpus_path = Path(corpus_dir)
    out_path    = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split_files = sorted(
        (p for p in corpus_path.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    print(f"Found {len(split_files)} corpus splits in {corpus_dir}")
    print(f"Writing TREC documents to {output_dir}")
    print()

    # Bucket documents by collection prefix for separate output files
    writers: dict[str, object] = {}

    def get_writer(prefix: str):
        if prefix not in writers:
            path = out_path / f"{prefix}.trec"
            writers[prefix] = open(path, "wb")
        return writers[prefix]

    def doc_prefix(name: str) -> str:
        for p in ("FBIS", "FT", "LA", "FR94", "FR9"):
            if name.startswith(p):
                return p.rstrip("9").rstrip("4") if p.startswith("FR") else p
        return "OTHER"

    total_docs = 0
    t0 = time.perf_counter()

    for split in split_files:
        t1 = time.perf_counter()
        docs_in_split = 0
        for name, content in iter_documents(str(split)):
            prefix = doc_prefix(name)
            w = get_writer(prefix)
            w.write(b"<DOC>\n<DOCNO> ")
            w.write(name.encode())
            w.write(b" </DOCNO>\n")
            w.write(content)
            if not content.endswith(b"\n"):
                w.write(b"\n")
            w.write(b"</DOC>\n")
            docs_in_split += 1

        total_docs += docs_in_split
        elapsed = time.perf_counter() - t0
        rate = total_docs / elapsed
        print(f"  split {split.name:3s}: {docs_in_split:6,} docs  |  "
              f"total {total_docs:,}  ({rate:.0f} docs/s)")

    for w in writers.values():
        w.close()

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {total_docs:,} documents extracted in {elapsed:.1f}s")
    print("Output files:")
    for p in sorted(out_path.iterdir()):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    CORPUS_DIR = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/corpus"
    OUTPUT_DIR = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-trec"
    extract_corpus(CORPUS_DIR, OUTPUT_DIR)

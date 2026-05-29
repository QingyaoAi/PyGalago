#!/usr/bin/env python3
"""PyGalago performance benchmark — Phase 8.

Measures:
  1. Index build time (small synthetic collection)
  2. Single-term BM25 query latency (if an index is available)
  3. Multi-term query throughput

Usage::

    python scripts/benchmark.py [--index DIR] [--queries N]
"""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import textwrap
import time


def _build_synthetic_collection(path: str, n_docs: int = 1000) -> None:
    words = [
        "information", "retrieval", "search", "document", "query",
        "ranking", "index", "relevance", "system", "model",
        "term", "frequency", "inverted", "score", "engine",
    ]
    import random
    rng = random.Random(42)
    with open(path, "w") as f:
        for i in range(n_docs):
            doc_words = " ".join(rng.choices(words, k=rng.randint(20, 100)))
            f.write(f"<DOC>\n<DOCNO> BENCH{i:06d} </DOCNO>\n<TEXT>\n{doc_words}\n</TEXT>\n</DOC>\n")


def bench_index_build(n_docs: int = 1000) -> dict:
    from pygalago.index.builder import IndexBuilder

    with tempfile.TemporaryDirectory() as td:
        col_path = os.path.join(td, "col.trec")
        idx_path  = os.path.join(td, "idx")

        _build_synthetic_collection(col_path, n_docs)

        t0 = time.perf_counter()
        with IndexBuilder(idx_path, stemmer="none", also_unstemmed=True) as b:
            count = b.add_documents_from_file(col_path)
        elapsed = time.perf_counter() - t0

        return {
            "docs": count,
            "elapsed_s": elapsed,
            "docs_per_sec": count / elapsed,
        }


def bench_query_latency(index_path: str, n_queries: int = 100) -> dict:
    from pygalago.retrieval import Retrieval

    queries = [
        "information retrieval",
        "search engine ranking",
        "BM25 document score",
        "inverted index term",
        "relevance model query",
    ]

    r = Retrieval(index_path)

    latencies = []
    for i in range(n_queries):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        results = r.search(q, n=1000)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

    return {
        "n_queries": n_queries,
        "mean_s":    statistics.mean(latencies),
        "median_s":  statistics.median(latencies),
        "p95_s":     sorted(latencies)[int(0.95 * len(latencies))],
        "min_s":     min(latencies),
        "max_s":     max(latencies),
        "qps":       n_queries / sum(latencies),
    }


def _fmt(label: str, val: float, unit: str = "") -> str:
    return f"  {label:<30s} {val:>10.3f} {unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description="PyGalago performance benchmark")
    parser.add_argument("--index",   metavar="DIR",
                        default=os.environ.get("GALAGO_INDEX_PATH", ""),
                        help="Index directory for query benchmarks")
    parser.add_argument("--n-docs",  type=int, default=10_000,
                        help="Docs for index build benchmark (default: 10000)")
    parser.add_argument("--n-queries", type=int, default=100,
                        help="Queries for latency benchmark (default: 100)")
    args = parser.parse_args()

    try:
        import pygalago._galago  # noqa: F401
    except ImportError:
        print("ERROR: C++ extension not built. Run `pip install -e .`")
        raise SystemExit(1)

    # ── Index build benchmark ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Index build benchmark  ({args.n_docs} synthetic docs)")
    print('='*50)
    res = bench_index_build(args.n_docs)
    print(_fmt("Documents indexed",       res["docs"]))
    print(_fmt("Total time",              res["elapsed_s"],  "s"))
    print(_fmt("Throughput",              res["docs_per_sec"], "docs/s"))

    # ── Query latency benchmark ────────────────────────────────────────────────
    if args.index and os.path.isdir(args.index):
        print(f"\n{'='*50}")
        print(f"Query latency benchmark  ({args.n_queries} queries, n=1000)")
        print(f"Index: {args.index}")
        print('='*50)
        res = bench_query_latency(args.index, args.n_queries)
        print(_fmt("Queries run",          res["n_queries"]))
        print(_fmt("Mean latency",         res["mean_s"] * 1000,   "ms"))
        print(_fmt("Median latency",       res["median_s"] * 1000, "ms"))
        print(_fmt("P95 latency",          res["p95_s"] * 1000,    "ms"))
        print(_fmt("Min / Max",
                   res["min_s"] * 1000,
                   f"ms / {res['max_s']*1000:.1f} ms"))
        print(_fmt("Throughput",           res["qps"],              "queries/s"))
    else:
        print("\n(Skipping query latency — no index path provided)")
        print("  Set GALAGO_INDEX_PATH or pass --index DIR")

    print()


if __name__ == "__main__":
    main()

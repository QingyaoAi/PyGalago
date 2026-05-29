"""Entry point for the `pygalago` CLI command."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pygalago",
        description="PyGalago — Galago search engine (Python + C++ port)",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── build-index ────────────────────────────────────────────────────────────
    p_build = sub.add_parser(
        "build-index",
        help="Build a Galago-format index from a document collection.",
    )
    p_build.add_argument("collection", help="Path to collection file(s)")
    p_build.add_argument("--index",    required=True, metavar="DIR",
                         help="Output index directory")
    p_build.add_argument("--stemmer",  default="krovetz",
                         choices=["krovetz", "porter", "none"],
                         help="Stemmer to use (default: krovetz)")
    p_build.add_argument("--no-unstemmed", action="store_true",
                         help="Skip writing the unstemmed postings part")
    p_build.add_argument("--chunk-size", type=int, default=100_000,
                         metavar="N",
                         help="Documents per in-memory chunk (default: 100000)")
    p_build.set_defaults(func=_cmd_build_index)

    # ── search ─────────────────────────────────────────────────────────────────
    p_search = sub.add_parser(
        "search",
        help="Run a query against an index and print ranked results.",
    )
    p_search.add_argument("--index", required=True, metavar="DIR",
                          help="Index directory")
    p_search.add_argument("--query", required=True,
                          help="Query string (Galago structured query language)")
    p_search.add_argument("-n", "--count", type=int, default=10,
                          help="Number of results to return (default: 10)")
    p_search.add_argument("--stemmer", default="krovetz",
                          choices=["krovetz", "porter", "none"])
    p_search.add_argument("-b", type=float, default=0.75,
                          help="BM25 b parameter (default: 0.75)")
    p_search.add_argument("-k", type=float, default=1.2,
                          help="BM25 k parameter (default: 1.2)")
    p_search.set_defaults(func=_cmd_search)

    # ── batch-search ───────────────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch-search",
        help="Run a query file and write a TREC run file.",
    )
    p_batch.add_argument("--index",   required=True, metavar="DIR")
    p_batch.add_argument("--queries", required=True, metavar="FILE",
                         help="Query file: one 'topic_id\\tquery' per line")
    p_batch.add_argument("--output",  required=True, metavar="FILE",
                         help="Output TREC run file")
    p_batch.add_argument("--run-tag", default="pygalago",
                         help="Run tag for the TREC run file (default: pygalago)")
    p_batch.add_argument("-n", "--count", type=int, default=1000)
    p_batch.add_argument("--stemmer", default="krovetz",
                         choices=["krovetz", "porter", "none"])
    p_batch.add_argument("-b", type=float, default=0.75)
    p_batch.add_argument("-k", type=float, default=1.2)
    p_batch.set_defaults(func=_cmd_batch_search)

    # ── dump-index ─────────────────────────────────────────────────────────────
    p_dump = sub.add_parser(
        "dump-index",
        help="Dump human-readable contents of an index part.",
    )
    p_dump.add_argument("--index", required=True, metavar="DIR")
    p_dump.add_argument("--part",  default="postings.krovetz",
                        help="Index part to dump (default: postings.krovetz)")
    p_dump.add_argument("--max-terms", type=int, default=0,
                        help="Maximum terms to show (0 = all)")
    p_dump.set_defaults(func=_cmd_dump_index)

    # ── eval ───────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "eval",
        help="Evaluate a TREC run file against qrels.",
    )
    p_eval.add_argument("--qrels",   required=True, metavar="FILE")
    p_eval.add_argument("--results", required=True, metavar="FILE",
                        help="TREC run file to evaluate")
    p_eval.add_argument("--metrics", nargs="+",
                        default=["map", "ndcg@10", "ndcg@20", "p@10", "mrr", "bpref"],
                        metavar="METRIC",
                        help="Metrics to compute (default: map ndcg@10 ndcg@20 p@10 mrr bpref)")
    p_eval.add_argument("--per-topic", action="store_true",
                        help="Print per-topic scores in addition to means")
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args()
    args.func(args)


# ── Command implementations ───────────────────────────────────────────────────

def _cmd_build_index(args: argparse.Namespace) -> None:
    from pygalago.index.builder import IndexBuilder

    print(f"Building index from {args.collection!r} → {args.index!r}")
    print(f"  stemmer={args.stemmer}  unstemmed={not args.no_unstemmed}")

    with IndexBuilder(
        args.index,
        stemmer=args.stemmer,
        also_unstemmed=not args.no_unstemmed,
        chunk_size=args.chunk_size,
    ) as builder:
        count = builder.add_documents_from_file(args.collection)

    print(f"Indexed {count} documents.")


def _cmd_search(args: argparse.Namespace) -> None:
    from pygalago.retrieval import Retrieval

    part = f"postings.{args.stemmer}" if args.stemmer != "none" else "postings"
    r = Retrieval(args.index, b=args.b, k=args.k, part=part)
    results = r.search(args.query, n=args.count)

    print(f"Query: {args.query!r}  (top {args.count} results)")
    print()
    for rank, (name, score) in enumerate(results, 1):
        print(f"  {rank:4d}  {score:10.6f}  {name}")


def _cmd_batch_search(args: argparse.Namespace) -> None:
    from pygalago.retrieval import Retrieval
    from pygalago.eval.run  import write_run, Run, RankedDoc

    part = f"postings.{args.stemmer}" if args.stemmer != "none" else "postings"
    r = Retrieval(args.index, b=args.b, k=args.k, part=part)

    run: Run = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            topic, query = parts[0].strip(), parts[1].strip()

            scored = r.search(query, n=args.count)
            run[topic] = [
                RankedDoc(doc_id=name, score=score, rank=i)
                for i, (name, score) in enumerate(scored, 1)
            ]
            print(f"  {topic}: {len(run[topic])} results", file=sys.stderr)

    write_run(run, args.output, run_tag=args.run_tag)
    print(f"Run written to {args.output!r}  ({len(run)} topics).")


def _cmd_dump_index(args: argparse.Namespace) -> None:
    import os

    try:
        import pygalago._galago as _g
    except ImportError:
        print("Error: C++ extension not built. Run `pip install -e .`", file=sys.stderr)
        sys.exit(1)

    part_path = os.path.join(args.index, args.part)
    if not os.path.exists(part_path):
        print(f"Error: part not found: {part_path!r}", file=sys.stderr)
        sys.exit(1)

    reader = _g.BTreeReader(part_path)
    print(f"Manifest: {reader.manifest_json}")
    print()

    it = reader.iterator()
    if it is None:
        print("(empty index)")
        return

    count = 0
    while not it.is_done:
        key   = it.key
        vlen  = it.value_length
        print(f"  {key!r:40s}  value_len={vlen}")
        it.next_key()
        count += 1
        if args.max_terms > 0 and count >= args.max_terms:
            print(f"  ... (truncated at {args.max_terms})")
            break

    print(f"\n{count} entries.")


def _cmd_eval(args: argparse.Namespace) -> None:
    from pygalago.eval import read_qrels, read_run, evaluate
    from pygalago.eval.metrics import (
        average_precision, ndcg_at_k, reciprocal_rank,
        precision_at_k, r_precision, bpref as _bpref,
        recall_at_k,
    )

    qrels = read_qrels(args.qrels)
    run   = read_run(args.results)

    ranked: dict[str, list[str]] = {
        t: [rd.doc_id for rd in docs]
        for t, docs in run.items()
    }

    scores = evaluate(ranked, qrels, metrics=args.metrics)

    # Print aggregate scores
    print(f"{'Metric':<20s}  {'Score':>10s}")
    print("-" * 33)
    for m in args.metrics:
        v = scores.get(m, 0.0)
        print(f"{m:<20s}  {v:10.4f}")

    if args.per_topic:
        print()
        _print_per_topic(ranked, qrels, args.metrics)


def _print_per_topic(
    ranked: dict,
    qrels: dict,
    metrics: list[str],
) -> None:
    from pygalago.eval.metrics import (
        average_precision, ndcg_at_k, precision_at_k,
        reciprocal_rank, bpref as _bpref,
    )
    from pygalago.eval.qrels import relevant_docs

    header = f"{'Topic':<15s}" + "".join(f"  {m:>10s}" for m in metrics)
    print(header)
    print("-" * len(header))

    for topic in sorted(ranked):
        rel = relevant_docs(qrels, topic)
        grades = qrels.get(topic, {})
        row = f"{topic:<15s}"
        for m in metrics:
            nl = m.lower()
            r  = ranked.get(topic, [])
            if nl == "map":
                v = average_precision(r, rel)
            elif nl.startswith("ndcg@"):
                k = int(nl[5:])
                v = ndcg_at_k(r, grades, k)
            elif nl.startswith("p@"):
                k = int(nl[2:])
                v = precision_at_k(r, rel, k)
            elif nl == "mrr":
                v = reciprocal_rank(r, rel)
            elif nl == "bpref":
                v = _bpref(r, rel)
            else:
                v = 0.0
            row += f"  {v:10.4f}"
        print(row)

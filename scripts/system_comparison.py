#!/usr/bin/env python3
"""System comparison: PyGalago vs Pyserini vs PyTerrier on Robust04.

Measures and reports:
  1. Corpus indexing speed
  2. Query runtime (per query, ms)
  3. Retrieval effectiveness: MAP, NDCG@20, P@20

Models compared: BM25, QL (Dirichlet, μ=2500), SDM (where available)

Usage
-----
    python scripts/system_comparison.py [--output paper/System\ Comparison.md]

Environment variables required (set before running):
    export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"
    export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
    export JVM_PATH="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────

CORPUS_DIR   = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-trec"
TITLES_TSV   = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/rob04.titles.tsv"
QRELS_FILE   = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/robust04.qrels"

PYGALAGO_IDX = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04.index"
PYSERINI_IDX = "/tmp/robust04-pyserini-index"
PYTERRIER_IDX = "/tmp/robust04-pyterrier-index"

# ── Model parameters ───────────────────────────────────────────────────────────

MU     = 2500.0
BM25_B = 0.75
BM25_K = 1.2
N      = 1000   # docs to retrieve per query

METRICS = ["map", "ndcg_cut_20", "P_20"]
METRIC_LABELS = {"map": "MAP", "ndcg_cut_20": "NDCG@20", "P_20": "P@20"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_queries(path: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
                qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def trec_eval(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]],
              metrics: List[str]) -> Dict[str, float]:
    """Simple trec_eval using pytrec_eval."""
    import pytrec_eval
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        set(metrics),
        relevance_level=1,
    )
    run_scores = {
        qid: {docid: float(N - rank) for rank, docid in enumerate(docs)}
        for qid, docs in run.items()
    }
    per_query = evaluator.evaluate(run_scores)
    return {m: sum(v[m] for v in per_query.values()) / len(per_query) for m in metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pyserini
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_trec_corpus(corpus_dir: str):
    """Yield (docno, text) tuples by parsing all *.trec files in corpus_dir."""
    import re
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".trec"):
            continue
        with open(os.path.join(corpus_dir, fname), encoding="utf-8", errors="replace") as f:
            content = f.read()
        for doc_match in re.finditer(r"<DOC>(.*?)</DOC>", content, re.DOTALL):
            doc = doc_match.group(1)
            m = re.search(r"<DOCNO>(.*?)</DOCNO>", doc)
            if not m:
                continue
            docno = m.group(1).strip()
            text_parts = re.findall(r"<(?:TEXT|HEADLINE)>(.*?)</(?:TEXT|HEADLINE)>", doc, re.DOTALL)
            text = " ".join(text_parts)
            yield docno, text


def run_pyserini(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> Dict:
    """Build Lucene index from TREC files, run BM25 and QL queries, evaluate."""
    print("\n" + "=" * 60)
    print("PYSERINI")
    print("=" * 60)

    from pyserini.index.lucene import LuceneIndexer
    from pyserini.search.lucene import LuceneSearcher

    results = {}

    # ── Indexing (parse TREC files → LuceneIndexer.add_batch_dict) ────────────
    if os.path.isdir(PYSERINI_IDX):
        shutil.rmtree(PYSERINI_IDX)

    print(f"Indexing {CORPUS_DIR} …", flush=True)
    t0 = time.perf_counter()
    indexer = LuceneIndexer(index_dir=PYSERINI_IDX, threads=4)
    batch: List[Dict] = []
    total_docs = 0
    for docno, text in _parse_trec_corpus(CORPUS_DIR):
        batch.append({"id": docno, "contents": text})
        total_docs += 1
        if len(batch) >= 1000:
            indexer.add_batch_dict(batch)
            batch.clear()
    if batch:
        indexer.add_batch_dict(batch)
    indexer.close()
    t_index = time.perf_counter() - t0
    print(f"  Indexed {total_docs:,} docs in {t_index:.1f}s")
    results["index_time"] = t_index

    searcher = LuceneSearcher(PYSERINI_IDX)
    n_queries = len(queries)

    def _run_model(name: str, setup_fn=None, teardown_fn=None):
        if setup_fn:
            setup_fn(searcher)
        t0 = time.perf_counter()
        run: Dict[str, List[str]] = {}
        for qid, text in queries.items():
            hits = searcher.search(text, k=N)
            run[qid] = [h.docid for h in hits]
        elapsed = time.perf_counter() - t0
        if teardown_fn:
            teardown_fn(searcher)
        metrics = trec_eval(run, qrels, METRICS)
        print(f"  {name:<6} MAP={metrics['map']:.4f}  NDCG@20={metrics['ndcg_cut_20']:.4f}  "
              f"P@20={metrics['P_20']:.4f}  ({elapsed/n_queries*1000:.0f}ms/q)")
        return elapsed, metrics

    # BM25
    results["BM25"] = {}
    results["BM25"]["time"], results["BM25"]["metrics"] = _run_model(
        "BM25",
        setup_fn=lambda s: s.set_bm25(b=BM25_B, k1=BM25_K),
    )

    # QL (Dirichlet)
    results["QL"] = {}
    results["QL"]["time"], results["QL"]["metrics"] = _run_model(
        "QL",
        setup_fn=lambda s: s.set_qld(mu=int(MU)),
    )

    results["n_queries"] = n_queries
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PyTerrier
# ═══════════════════════════════════════════════════════════════════════════════

def run_pyterrier(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> Dict:
    """Build index, run BM25, QL, and SDM queries, evaluate."""
    print("\n" + "=" * 60)
    print("PYTERRIER")
    print("=" * 60)

    import pyterrier as pt
    pt.java.init()

    results = {}
    n_queries = len(queries)

    # ── Indexing ──────────────────────────────────────────────────────────────
    if os.path.isdir(PYTERRIER_IDX):
        shutil.rmtree(PYTERRIER_IDX)

    print(f"Indexing {CORPUS_DIR} …", end=" ", flush=True)
    t0 = time.perf_counter()

    indexer = pt.TRECCollectionIndexer(
        PYTERRIER_IDX,
        stemmer=pt.TerrierStemmer.porter,
        stopwords=pt.TerrierStopwords.terrier,
        tokeniser=pt.TerrierTokeniser.english,
        blocks=True,          # needed for SDM/proximity
        overwrite=True,
    )
    indexref = indexer.index(
        [os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR)
         if f.endswith(".trec")]
    )
    t_index = time.perf_counter() - t0
    print(f"done ({t_index:.1f}s)")
    results["index_time"] = t_index

    # ── Build query dataframe ──────────────────────────────────────────────────
    import pandas as pd
    topics_df = pd.DataFrame(
        [{"qid": qid, "query": text} for qid, text in queries.items()]
    )

    def _eval_run(run_df: pd.DataFrame) -> Dict[str, float]:
        qrels_rows = []
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                qrels_rows.append({"qid": qid, "docno": docid, "label": rel})
        qrels_df = pd.DataFrame(qrels_rows)
        # Compute metrics manually using pytrec_eval
        run_dict: Dict[str, List[str]] = {}
        for _, row in run_df.iterrows():
            run_dict.setdefault(str(row["qid"]), []).append(str(row["docno"]))
        return trec_eval(run_dict, qrels, METRICS)

    def _run_model(name: str, retriever):
        t0 = time.perf_counter()
        run_df = retriever.transform(topics_df)
        elapsed = time.perf_counter() - t0
        metrics = _eval_run(run_df)
        print(f"  {name:<6} MAP={metrics['map']:.4f}  NDCG@20={metrics['ndcg_cut_20']:.4f}  "
              f"P@20={metrics['P_20']:.4f}  ({elapsed/n_queries*1000:.0f}ms/q)")
        return elapsed, metrics

    # BM25
    bm25 = pt.BatchRetrieve(
        indexref,
        wmodel="BM25",
        num_results=N,
        controls={"b": BM25_B, "k_1": BM25_K, "k_3": 0},
    )
    results["BM25"] = {}
    results["BM25"]["time"], results["BM25"]["metrics"] = _run_model("BM25", bm25)

    # QL / Dirichlet Language Model
    ql = pt.BatchRetrieve(
        indexref,
        wmodel="DirichletLM",
        num_results=N,
        controls={"c": MU},
    )
    results["QL"] = {}
    results["QL"]["time"], results["QL"]["metrics"] = _run_model("QL", ql)

    # SDM — Sequential Dependence Model
    sdm = pt.rewrite.SDM() >> pt.BatchRetrieve(
        indexref,
        wmodel="DirichletLM",
        num_results=N,
        controls={"c": MU},
    )
    results["SDM"] = {}
    results["SDM"]["time"], results["SDM"]["metrics"] = _run_model("SDM", sdm)

    results["n_queries"] = n_queries
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PyGalago — read from existing results
# ═══════════════════════════════════════════════════════════════════════════════

def run_pygalago(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> Dict:
    """Run PyGalago BM25, QL, SDM fresh and return timing + metrics."""
    print("\n" + "=" * 60)
    print("PYGALAGO")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pygalago._galago as g
    from pygalago.parse.stemmer   import get_stemmer
    from pygalago.parse.tokenizer import tokenize_string
    from pygalago.retrieval       import Retrieval

    PART         = "postings.porter"
    STEMMER_NAME = "porter"

    INQUERY_STOPS = frozenset({
        "a","about","above","according","across","after","afterwards","again","against",
        "albeit","all","almost","alone","along","already","also","although","always","am",
        "among","amongst","an","and","another","any","anybody","anyhow","anyone","anything",
        "anyway","anywhere","apart","are","around","as","at","av","be","became","because",
        "become","becomes","becoming","been","before","beforehand","behind","being","below",
        "beside","besides","between","beyond","both","but","by","can","cannot","canst",
        "certain","cf","choose","contrariwise","cos","could","couldn't","dare","daren't",
        "definitely","despite","did","didn't","different","directly","do","does","doesn't",
        "doing","done","don't","down","during","e","each","eg","either","else","elsewhere",
        "enough","etc","even","ever","every","everybody","everyone","everything","everywhere",
        "except","exactly","far","few","ff","fifth","first","following","for","former",
        "formerly","forth","from","further","furthermore","get","given","go","got","h",
        "had","hadn't","has","hasn't","have","haven't","having","he","her","here",
        "hereabouts","hereafter","hereby","herein","hereinafter","heretofore","hereunder",
        "hereupon","herewith","him","himself","his","how","however","i","ie","if","in",
        "indeed","inside","instead","into","is","isn't","it","its","itself","just","kind",
        "kg","km","last","latter","latterly","less","lest","let","like","little","lots",
        "many","may","maybe","me","meantime","meanwhile","might","moreover","most","mostly",
        "more","mr","mrs","much","my","myself","namely","needn't","neither","never",
        "nevertheless","next","no","nobody","none","noone","nothing","notwithstanding","now",
        "nowhere","of","off","often","ok","on","once","one","only","onto","or","other",
        "others","otherwise","ought","our","ours","ourselves","out","outside","over","own",
        "per","perhaps","please","rather","re","really","regarding","same","sans","self",
        "several","should","shouldn't","since","so","some","somebody","somehow","someone",
        "something","sometime","sometimes","somewhere","still","such","than","that","the",
        "thee","their","theirs","them","themselves","then","thence","there","thereabouts",
        "thereafter","thereby","therfore","therefore","therein","these","they","this",
        "those","thou","though","through","throughout","thru","thus","thy","till","to",
        "together","too","toward","towards","under","unless","until","up","upon","us",
        "very","via","vs","was","wasn't","we","were","weren't","what","whatever","when",
        "whence","whenever","where","whereabouts","whereas","whereby","whether","which",
        "while","whither","who","whoever","whom","whomsoever","whose","why","will","with",
        "within","without","won't","would","wouldn't","you","your","yours","yourself",
        "yourselves",
    })

    import bisect, math
    from array import array as _array
    from collections import defaultdict

    print("Loading PyGalago index …", end=" ", flush=True)
    t0 = time.perf_counter()
    idx   = g.DiskIndex(PYGALAGO_IDX)
    pr    = g.PostingsReader(os.path.join(PYGALAGO_IDX, PART))
    ls    = g.LengthsSource(os.path.join(PYGALAGO_IDX, "lengths"))
    stats = ls.stats
    C     = stats.collection_length
    print(f"{stats.total_document_count:,} docs ({time.perf_counter()-t0:.1f}s)")

    stem = get_stemmer(STEMMER_NAME)

    def process(text):
        tokens = tokenize_string(text)
        return [stem(t) for t in tokens if t and t not in INQUERY_STOPS]

    def _ql_ids(terms, n=N) -> list:
        """Return List[Tuple[int, float]] (doc_id, score)."""
        info = []
        for t in dict.fromkeys(terms):
            s = pr.get_stats(t)
            if s is None: continue
            it = pr.get_postings(t)
            if it is None: continue
            info.append((s["collection_count"] / C, it))
        if not info: return []
        log_mu_pt = [math.log(MU * p) for p, _ in info]
        slmp = math.fsum(log_mu_pt)
        corr: Dict[int, float] = defaultdict(float)
        for (p_t, it), lmp in zip(info, log_mu_pt):
            for d, tf in it:
                corr[d] += math.log(tf + MU * p_t) - lmp
        scored = [(d, slmp + c - len(info) * math.log(ls.length(d) + MU))
                  for d, c in corr.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def _sdm_ids(terms) -> list:
        """Return List[Tuple[int, float]] (doc_id, score) using 2-stage SDM."""
        terms = list(dict.fromkeys(terms))
        if len(terms) < 2:
            return _ql_ids(terms)
        cands = _ql_ids(terms, n=5000)
        if not cands: return []
        cand_ids = sorted(d for d, _ in cands)

        term_stats = []
        for t in terms:
            s = pr.get_stats(t)
            p_t = s["collection_count"] / C if s else 0.0
            pos_data = pr.read_positions_for(t, cand_ids)
            term_stats.append((p_t, {d: ps for d, ps in pos_data}))

        scored = []
        for d in cand_ids:
            dl = ls.length(d)
            denom = dl + MU
            uni = sum(math.log(max(len(dp.get(d, [])) + MU * p_t, 1e-300) / denom)
                      for p_t, dp in term_stats)
            od = uw = 0.0
            for i in range(len(terms) - 1):
                p1 = term_stats[i][1].get(d, [])
                p2 = term_stats[i + 1][1].get(d, [])
                s2 = set(p2)
                od_tf = sum(1 for p in p1 if (p + 1) in s2)
                uw_tf = 0
                for p in p1:
                    lo = bisect.bisect_left(p2, p - 7)
                    hi = bisect.bisect_right(p2, p + 7)
                    if lo < hi: uw_tf += 1
                cf = max(od_tf, 1) / C
                od += math.log((od_tf + MU * cf) / denom)
                uw += math.log((uw_tf + MU * cf) / denom)
            scored.append((d, 0.85 * uni + 0.10 * od + 0.05 * uw))
        scored.sort(key=lambda x: -x[1])
        return scored[:N]

    bm25_ret = Retrieval(PYGALAGO_IDX, b=BM25_B, k=BM25_K, part=PART, stemmer=STEMMER_NAME)

    results = {}
    n_queries = len(queries)

    def _run_model(name, search_fn):
        t0 = time.perf_counter()
        run: Dict[str, List[str]] = {}
        for qid, text in queries.items():
            run[qid] = [docid for docid, _ in search_fn(text)]
        elapsed = time.perf_counter() - t0
        metrics = trec_eval(run, qrels, METRICS)
        print(f"  {name:<6} MAP={metrics['map']:.4f}  NDCG@20={metrics['ndcg_cut_20']:.4f}  "
              f"P@20={metrics['P_20']:.4f}  ({elapsed/n_queries*1000:.0f}ms/q)")
        return elapsed, metrics

    results["BM25"] = {}
    results["BM25"]["time"], results["BM25"]["metrics"] = _run_model(
        "BM25",
        lambda text: bm25_ret.search(
            " ".join(t for t in tokenize_string(text) if t not in INQUERY_STOPS) or text,
            n=N
        ),
    )

    results["QL"] = {}
    results["QL"]["time"], results["QL"]["metrics"] = _run_model(
        "QL",
        lambda text: [(idx.get_name(d), s) for d, s in _ql_ids(process(text))],
    )

    results["SDM"] = {}
    results["SDM"]["time"], results["SDM"]["metrics"] = _run_model(
        "SDM",
        lambda text: [(idx.get_name(d), s) for d, s in _sdm_ids(process(text))],
    )

    results["n_queries"] = n_queries
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(pg_r, ps_r, pt_r) -> str:
    n = pg_r["n_queries"]
    lines: List[str] = []
    lines.append("## System Comparison: PyGalago vs Pyserini vs PyTerrier\n")
    lines.append("**Collection:** Robust04 (528,155 documents)  ")
    lines.append(f"**Queries:** {n} TREC title queries  ")
    lines.append("**Stemmer:** Porter2  **Stopping:** INQUERY (PyGalago), Terrier default (PyTerrier)\n")

    # ── Indexing ──────────────────────────────────────────────────────────────
    lines.append("\n### Indexing Time\n")
    lines.append("| System     | Time (s) |")
    lines.append("|------------|----------|")
    lines.append(f"| PyGalago   | — (pre-built) |")
    ps_idx = f"{ps_r['index_time']:.1f}" if ps_r.get('index_time') is not None else "—"
    pt_idx = f"{pt_r['index_time']:.1f}" if pt_r.get('index_time') is not None else "—"
    lines.append(f"| Pyserini   | {ps_idx} |")
    lines.append(f"| PyTerrier  | {pt_idx} |")

    # ── Retrieval effectiveness ────────────────────────────────────────────────
    lines.append("\n### Retrieval Effectiveness\n")
    lines.append("| System | Model | MAP | NDCG@20 | P@20 |")
    lines.append("|--------|-------|-----|---------|------|")

    for system, r, models in [
        ("PyGalago",  pg_r, ["BM25", "QL", "SDM"]),
        ("Pyserini",  ps_r, ["BM25", "QL"]),
        ("PyTerrier", pt_r, ["BM25", "QL", "SDM"]),
    ]:
        for m in models:
            if m not in r: continue
            me = r[m]["metrics"]
            lines.append(
                f"| {system} | {m} | {me['map']:.4f} | {me['ndcg_cut_20']:.4f} | {me['P_20']:.4f} |"
            )

    # ── Query runtime ─────────────────────────────────────────────────────────
    lines.append("\n### Query Runtime (ms/query)\n")
    lines.append("| System | Model | ms/query |")
    lines.append("|--------|-------|----------|")

    for system, r, models in [
        ("PyGalago",  pg_r, ["BM25", "QL", "SDM"]),
        ("Pyserini",  ps_r, ["BM25", "QL"]),
        ("PyTerrier", pt_r, ["BM25", "QL", "SDM"]),
    ]:
        nq = r["n_queries"]
        for m in models:
            if m not in r: continue
            ms_per_q = r[m]["time"] / nq * 1000
            lines.append(f"| {system} | {m} | {ms_per_q:.0f} |")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker entry points (each runs in its own subprocess to isolate the JVM)
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_pygalago():
    import json
    queries = load_queries(TITLES_TSV)
    qrels   = load_qrels(QRELS_FILE)
    r = run_pygalago(queries, qrels)
    # Serialize: replace non-serialisable fields
    out = {"index_time": None, "n_queries": r["n_queries"]}
    for m in ["BM25", "QL", "SDM"]:
        if m in r:
            out[m] = {"time": r[m]["time"], "metrics": r[m]["metrics"]}
    print("__RESULT__" + json.dumps(out))


def _worker_pyserini():
    import json
    queries = load_queries(TITLES_TSV)
    qrels   = load_qrels(QRELS_FILE)
    r = run_pyserini(queries, qrels)
    out = {"index_time": r["index_time"], "n_queries": r["n_queries"]}
    for m in ["BM25", "QL"]:
        if m in r:
            out[m] = {"time": r[m]["time"], "metrics": r[m]["metrics"]}
    print("__RESULT__" + json.dumps(out))


def _worker_pyterrier():
    import json
    queries = load_queries(TITLES_TSV)
    qrels   = load_qrels(QRELS_FILE)
    r = run_pyterrier(queries, qrels)
    out = {"index_time": r["index_time"], "n_queries": r["n_queries"]}
    for m in ["BM25", "QL", "SDM"]:
        if m in r:
            out[m] = {"time": r[m]["time"], "metrics": r[m]["metrics"]}
    print("__RESULT__" + json.dumps(out))


def _run_worker(mode: str) -> dict:
    """Launch this script as a subprocess with --worker=<mode>, return JSON result."""
    import json, subprocess
    env = os.environ.copy()
    env.setdefault("PATH", "/opt/homebrew/opt/openjdk@21/bin:" + os.environ.get("PATH", ""))
    env["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
    env["JVM_PATH"] = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib"

    cmd = [sys.executable, os.path.abspath(__file__), f"--worker={mode}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    result_line = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("__RESULT__"):
            result_line = line[len("__RESULT__"):]
        else:
            # Forward non-result lines to stdout, filtering noise
            skip = any(s in line for s in [
                "SLF4J", "DeprecationWarning", "pt.init", "pool-",
                "at io.anserini", "CompletionException", "EmptyDocument",
                "Caused by:", "Exception in thread", "MemorySegment",
                "WARNING: Using incubator", "SimpleIndexer",
                "terrier-assemblies", "terrier-python", "https://",
                "at java.", "at org.", "at com.",
            ])
            if not skip:
                print(line, flush=True)
    proc.wait()
    if result_line:
        return json.loads(result_line)
    raise RuntimeError(f"Worker {mode} did not produce a result (exit code {proc.returncode})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="System comparison on Robust04")
    ap.add_argument("--output", default="paper/System Comparison.md",
                    help="Output markdown file (default: paper/System Comparison.md)")
    ap.add_argument("--skip-pyserini", action="store_true")
    ap.add_argument("--skip-pyterrier", action="store_true")
    ap.add_argument("--worker", choices=["pygalago", "pyserini", "pyterrier"],
                    help="Internal: run as a worker process for one system")
    args = ap.parse_args()

    # Worker mode: run one system and print JSON result
    if args.worker == "pygalago":
        _worker_pygalago(); return
    if args.worker == "pyserini":
        _worker_pyserini(); return
    if args.worker == "pyterrier":
        _worker_pyterrier(); return

    # Orchestrator mode: launch workers as separate subprocesses
    print("Running PyGalago …")
    pg_r = _run_worker("pygalago")

    if args.skip_pyserini:
        ps_r = {"index_time": None, "n_queries": pg_r["n_queries"]}
    else:
        print("\nRunning Pyserini …")
        ps_r = _run_worker("pyserini")

    if args.skip_pyterrier:
        pt_r = {"index_time": None, "n_queries": pg_r["n_queries"]}
    else:
        print("\nRunning PyTerrier …")
        pt_r = _run_worker("pyterrier")

    report = build_report(pg_r, ps_r, pt_r)
    print("\n\n" + report)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("## Corpus\n")
        f.write(f"* {CORPUS_DIR}\n\n")
        f.write("## System to compare\n")
        f.write("pyserini: https://github.com/castorini/pyserini\n")
        f.write("pyterrier: https://github.com/terrier-org/pyterrier\n\n")
        f.write("## Things to compare\n")
        f.write("* Corpus indexing speed\n")
        f.write("* Query runtime\n")
        f.write("* performance of term-based methods like QL, BM25, SDM\n\n")
        f.write(report + "\n")
    print(f"\nResults written to {args.output!r}")


if __name__ == "__main__":
    main()

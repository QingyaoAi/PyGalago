# BSD License (http://www.galagosearch.org/license)
"""pygalago.eval — information retrieval evaluation metrics (Phase 6).

Usage::

    from pygalago.eval import read_qrels, read_run, evaluate

    qrels = read_qrels("qrels.robust04.txt")
    run   = read_run("my_run.txt")
    # convert to ranked lists
    ranked = {topic: [r.doc_id for r in docs] for topic, docs in run.items()}
    scores = evaluate(ranked, qrels)
    print(scores)
    # {'map': 0.254, 'ndcg@10': 0.421, ...}
"""

from pygalago.eval.qrels   import read_qrels, write_qrels, relevant_docs, Qrels
from pygalago.eval.run     import read_run, write_run, from_scored_documents, Run, RankedDoc
from pygalago.eval.metrics import (
    precision_at_k,
    recall_at_k,
    r_precision,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    mean_ndcg_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    bpref,
    evaluate,
)

__all__ = [
    # qrels
    "read_qrels", "write_qrels", "relevant_docs", "Qrels",
    # run
    "read_run", "write_run", "from_scored_documents", "Run", "RankedDoc",
    # metrics
    "precision_at_k", "recall_at_k", "r_precision",
    "average_precision", "mean_average_precision",
    "ndcg_at_k", "mean_ndcg_at_k",
    "reciprocal_rank", "mean_reciprocal_rank",
    "bpref",
    "evaluate",
]

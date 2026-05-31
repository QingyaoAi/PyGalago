# BSD License (http://www.galagosearch.org/license)
"""Port of Galago's eval/metric/ — standard IR evaluation metrics.

All functions take a ranked list of document IDs (strings) and a set of
relevant document IDs, and return a float.

Graded metrics (NDCG) accept a dict {doc_id: grade} for the qrels argument.

References
----------
- Voorhees & Harman (2005) TREC: Experiment and Evaluation in IR
- Manning, Raghavan, Schütze (2008) Introduction to Information Retrieval
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Set, Union

# Type aliases
DocList  = List[str]
RelSet   = Union[Set[str], frozenset]
GradedQrels = Dict[str, int]


# ── Precision / Recall ────────────────────────────────────────────────────────

def precision_at_k(ranked: DocList, relevant: RelSet, k: int) -> float:
    """Precision at rank k."""
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / k


def recall_at_k(ranked: DocList, relevant: RelSet, k: int) -> float:
    """Recall at rank k."""
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def r_precision(ranked: DocList, relevant: RelSet) -> float:
    """R-Precision: precision at rank |R| (number of relevant docs)."""
    r = len(relevant)
    if r == 0:
        return 0.0
    return precision_at_k(ranked, relevant, r)


# ── Average Precision ─────────────────────────────────────────────────────────

def average_precision(ranked: DocList, relevant: RelSet) -> float:
    """Average Precision (AP) for a single query.

    AP = (1/R) * Σ P@k * rel(k)   where the sum is over all ranks k.
    """
    if not relevant:
        return 0.0
    R = len(relevant)
    hits = 0
    ap   = 0.0
    for k, doc in enumerate(ranked, 1):
        if doc in relevant:
            hits += 1
            ap   += hits / k
    return ap / R


def mean_average_precision(
    ranked_lists: Dict[str, DocList],
    qrels: Dict[str, RelSet],
) -> float:
    """Mean Average Precision (MAP) over a set of topics."""
    aps: list[float] = []
    for topic, ranked in ranked_lists.items():
        rel = qrels.get(topic, frozenset())
        aps.append(average_precision(ranked, rel))
    return sum(aps) / len(aps) if aps else 0.0


# ── NDCG ──────────────────────────────────────────────────────────────────────

def _dcg(ranked: DocList, grades: GradedQrels, k: int) -> float:
    """Discounted Cumulative Gain at rank k (log base 2)."""
    dcg = 0.0
    for i, doc in enumerate(ranked[:k], 1):
        g = grades.get(doc, 0)
        if g > 0:
            dcg += (2 ** g - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    ranked: DocList,
    grades: GradedQrels,
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    Parameters
    ----------
    ranked:
        Ordered list of retrieved document IDs.
    grades:
        Dict mapping doc_id → relevance grade (integer ≥ 0).
        Documents not in the dict are treated as grade 0.
    k:
        Cutoff rank.
    """
    if not grades or k <= 0:
        return 0.0
    actual_dcg = _dcg(ranked, grades, k)
    # Ideal: sort by grade descending, take top k
    ideal_ranked = sorted(grades, key=lambda d: grades[d], reverse=True)
    ideal_dcg    = _dcg(ideal_ranked, grades, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def mean_ndcg_at_k(
    ranked_lists: Dict[str, DocList],
    graded_qrels: Dict[str, GradedQrels],
    k: int,
) -> float:
    """Mean NDCG@k over a set of topics."""
    scores: list[float] = []
    for topic, ranked in ranked_lists.items():
        grades = graded_qrels.get(topic, {})
        scores.append(ndcg_at_k(ranked, grades, k))
    return sum(scores) / len(scores) if scores else 0.0


# ── MRR ───────────────────────────────────────────────────────────────────────

def reciprocal_rank(ranked: DocList, relevant: RelSet) -> float:
    """Reciprocal Rank (RR) for a single query."""
    for k, doc in enumerate(ranked, 1):
        if doc in relevant:
            return 1.0 / k
    return 0.0


def mean_reciprocal_rank(
    ranked_lists: Dict[str, DocList],
    qrels: Dict[str, RelSet],
) -> float:
    """Mean Reciprocal Rank (MRR) over a set of topics."""
    rrs: list[float] = []
    for topic, ranked in ranked_lists.items():
        rel = qrels.get(topic, frozenset())
        rrs.append(reciprocal_rank(ranked, rel))
    return sum(rrs) / len(rrs) if rrs else 0.0


# ── Binary Preference (Bpref) ─────────────────────────────────────────────────

def bpref(ranked: DocList, relevant: RelSet, non_relevant: RelSet | None = None) -> float:
    """Binary Preference measure (Buckley & Voorhees 2004).

    Only judges are counted: relevant docs and (if supplied) known
    non-relevant docs.  When *non_relevant* is None, all unjudged docs in
    *ranked* are treated as non-relevant (standard approximation).

    bpref = (1/R) * Σ_r [ 1 - (#non-rel ranked above r) / R ]
    """
    R = len(relevant)
    if R == 0:
        return 0.0

    if non_relevant is None:
        # Use every unjudged retrieved doc as non-relevant
        non_relevant = frozenset(d for d in ranked if d not in relevant)

    nr_seen = 0
    score   = 0.0
    for doc in ranked:
        if doc in non_relevant:
            nr_seen += 1
        elif doc in relevant:
            score += 1.0 - min(nr_seen, R) / R
    return score / R


# ── Convenience: evaluate a full run against qrels ────────────────────────────

def evaluate(
    run: Dict[str, DocList],
    qrels: Dict[str, Dict[str, int]],
    *,
    metrics: Iterable[str] = ("map", "ndcg@10", "ndcg@20", "mrr", "p@10", "bpref"),
    min_grade: int = 1,
) -> Dict[str, float]:
    """Compute a dict of metric → score for a run against qrels.

    Parameters
    ----------
    run:
        ``{topic: [doc_id, ...]}`` ranked lists.
    qrels:
        ``{topic: {doc_id: grade}}`` relevance judgments.
    metrics:
        Which metrics to compute.  Supported names:
        ``map``, ``mrr``, ``bpref``, ``r-prec``,
        ``p@k`` (e.g. ``p@10``), ``ndcg@k`` (e.g. ``ndcg@20``).
    min_grade:
        Minimum grade to count as relevant for binary metrics.
    """
    metrics = list(metrics)
    rel_sets: Dict[str, frozenset] = {
        t: frozenset(d for d, g in q.items() if g >= min_grade)
        for t, q in qrels.items()
    }

    results: Dict[str, float] = {}

    for name in metrics:
        nl = name.lower()

        if nl == "map":
            results[name] = mean_average_precision(run, rel_sets)  # type: ignore[arg-type]

        elif nl == "mrr":
            results[name] = mean_reciprocal_rank(run, rel_sets)  # type: ignore[arg-type]

        elif nl in ("bpref", "binary_preference"):
            bprefs = [
                bpref(run.get(t, []), rel_sets.get(t, frozenset()))
                for t in run
            ]
            results[name] = sum(bprefs) / len(bprefs) if bprefs else 0.0

        elif nl in ("r-prec", "r_prec", "rprecision"):
            rps = [
                r_precision(run.get(t, []), rel_sets.get(t, frozenset()))
                for t in run
            ]
            results[name] = sum(rps) / len(rps) if rps else 0.0

        elif nl.startswith("p@"):
            k = int(nl[2:])
            p_scores = [
                precision_at_k(run.get(t, []), rel_sets.get(t, frozenset()), k)
                for t in run
            ]
            results[name] = sum(p_scores) / len(p_scores) if p_scores else 0.0

        elif nl.startswith("ndcg@"):
            k = int(nl[5:])
            graded = {t: dict(q) for t, q in qrels.items()}
            results[name] = mean_ndcg_at_k(run, graded, k)

        else:
            raise ValueError(f"Unknown metric: {name!r}")

    return results

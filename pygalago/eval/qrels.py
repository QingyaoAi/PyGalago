# BSD License (http://www.galagosearch.org/license)
"""Read and write TREC qrels files.

TREC qrels format (whitespace-separated)::

    <topic_id> <iteration> <doc_id> <relevance>

where ``<iteration>`` is conventionally ``0`` and ``<relevance>`` is an
integer (0 = not relevant, 1+ = relevant; some collections use ≥1 for
relevant, or graded 0/1/2/3 for graded relevance).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, IO, Iterable, Iterator, Tuple

# {topic_id: {doc_id: relevance_grade}}
Qrels = Dict[str, Dict[str, int]]


def read_qrels(source: str | IO) -> Qrels:
    """Parse a TREC qrels file and return a nested dict.

    Parameters
    ----------
    source:
        Path to the qrels file, or an open text stream.
    """
    qrels: Qrels = defaultdict(dict)

    def _parse(stream: IO) -> None:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            topic, _, doc_id, grade = parts[0], parts[1], parts[2], parts[3]
            qrels[topic][doc_id] = int(grade)

    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            _parse(f)
    else:
        _parse(source)

    return dict(qrels)


def write_qrels(qrels: Qrels, dest: str | IO, iteration: str = "0") -> None:
    """Write *qrels* to *dest* in TREC format."""
    def _write(stream: IO) -> None:
        for topic in sorted(qrels):
            for doc_id in sorted(qrels[topic]):
                grade = qrels[topic][doc_id]
                stream.write(f"{topic} {iteration} {doc_id} {grade}\n")

    if isinstance(dest, str):
        with open(dest, "w", encoding="utf-8") as f:
            _write(f)
    else:
        _write(dest)


def relevant_docs(qrels: Qrels, topic: str, min_grade: int = 1) -> frozenset:
    """Return the set of relevant doc IDs for *topic*."""
    return frozenset(
        doc for doc, grade in qrels.get(topic, {}).items()
        if grade >= min_grade
    )

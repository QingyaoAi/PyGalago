# BSD License (http://www.galagosearch.org/license)
"""Read and write TREC run files.

TREC run format (whitespace-separated, 6 columns)::

    <topic_id> Q0 <doc_id> <rank> <score> <run_tag>

where ``Q0`` is a literal field, ``<rank>`` starts at 1, and ``<score>``
is a floating-point relevance score (descending order assumed).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, IO, List, NamedTuple, Tuple


class RankedDoc(NamedTuple):
    doc_id: str
    score: float
    rank: int


# {topic_id: [RankedDoc, ...]} sorted by descending score
Run = Dict[str, List[RankedDoc]]


def read_run(source: str | IO) -> Run:
    """Parse a TREC run file and return a dict of ranked lists.

    Parameters
    ----------
    source:
        Path to the run file, or an open text stream.
    """
    raw: Dict[str, List[Tuple[float, int, str]]] = defaultdict(list)

    def _parse(stream: IO) -> None:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            topic = parts[0]
            doc_id = parts[2]
            rank   = int(parts[3])
            score  = float(parts[4])
            raw[topic].append((score, rank, doc_id))

    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            _parse(f)
    else:
        _parse(source)

    run: Run = {}
    for topic, entries in raw.items():
        entries.sort(key=lambda x: (-x[0], x[1]))  # descending score
        run[topic] = [RankedDoc(doc_id=d, score=s, rank=r)
                      for s, r, d in entries]
    return run


def write_run(run: Run, dest: str | IO, run_tag: str = "pygalago") -> None:
    """Write *run* to *dest* in TREC 6-column format."""
    def _write(stream: IO) -> None:
        for topic in sorted(run, key=lambda t: (t.zfill(10) if t.isdigit() else t)):
            for rank, rdoc in enumerate(run[topic], 1):
                stream.write(
                    f"{topic} Q0 {rdoc.doc_id} {rank} {rdoc.score:.6f} {run_tag}\n"
                )

    if isinstance(dest, str):
        with open(dest, "w", encoding="utf-8") as f:
            _write(f)
    else:
        _write(dest)


def from_scored_documents(
    topic: str,
    scored: list,
    *,
    name_fn=None,
) -> Run:
    """Convert a list of ScoredDocument (from C++ or (name, score) tuples) to a Run.

    Parameters
    ----------
    scored:
        Either a list of ``(name, score)`` tuples (from ``Retrieval.search``),
        or a list of C++ ``ScoredDocument`` objects with ``.document`` (int)
        and ``.score`` attributes when *name_fn* is provided.
    name_fn:
        Optional callable ``docid → name`` used when *scored* contains
        integer docids rather than string names.
    """
    entries: list[RankedDoc] = []
    for rank, item in enumerate(scored, 1):
        if isinstance(item, tuple):
            name, score = item
        else:
            name  = name_fn(item.document) if name_fn else str(item.document)
            score = item.score
        entries.append(RankedDoc(doc_id=name, score=score, rank=rank))
    return {topic: entries}

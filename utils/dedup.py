"""Deduplication utilities: ROUGE-based semantic dedup and exact dedup."""
from typing import List, Tuple, Callable, Optional
from functools import partial
from multiprocessing import Pool

from rouge_score import rouge_scorer


def create_rouge_scorer(
    metrics: Optional[List[str]] = None,
    use_stemmer: bool = False,
) -> rouge_scorer.RougeScorer:
    """Create a reusable RougeScorer instance. Default metric: rougeL."""
    if metrics is None:
        metrics = ["rougeL"]
    return rouge_scorer.RougeScorer(metrics, use_stemmer=use_stemmer)


def _score_pair(scorer, metric, candidate, reference):
    """Helper for multiprocessing: compute a single ROUGE score."""
    scores = scorer.score(candidate, reference)
    return scores[metric].fmeasure


def compute_rouge_scores(
    candidate: str,
    references: List[str],
    scorer: rouge_scorer.RougeScorer,
    metric: str = "rougeL",
    num_workers: int = 4,
) -> List[float]:
    """
    Compute ROUGE fmeasure between candidate and all references.
    Uses multiprocessing Pool for parallelism on large reference sets.
    Falls back to sequential for small sets.
    """
    if not references:
        return []

    if len(references) < 50:
        return [
            scorer.score(candidate, ref)[metric].fmeasure
            for ref in references
        ]

    fn = partial(_score_pair, scorer, metric, candidate)
    with Pool(num_workers) as pool:
        scores = pool.map(fn, references)
    return scores


def is_duplicate(
    candidate: str,
    references: List[str],
    scorer: rouge_scorer.RougeScorer,
    threshold: float = 0.7,
    num_workers: int = 4,
) -> bool:
    """Return True if candidate is too similar (above threshold) to any reference."""
    if not references:
        return False
    scores = compute_rouge_scores(candidate, references, scorer, num_workers=num_workers)
    return max(scores) > threshold


def deduplicate_by_rouge(
    candidates: List[str],
    existing: Optional[List[str]] = None,
    threshold: float = 0.7,
    num_workers: int = 4,
) -> Tuple[List[str], List[int]]:
    """
    Filter candidates against existing pool + each other.
    Returns (kept_candidates, kept_indices).
    Incrementally adds kept candidates to the reference pool.
    """
    if existing is None:
        existing = []
    pool = list(existing)
    scorer = create_rouge_scorer()
    kept = []
    kept_indices = []

    for i, candidate in enumerate(candidates):
        if not is_duplicate(candidate, pool, scorer, threshold, num_workers):
            kept.append(candidate)
            kept_indices.append(i)
            pool.append(candidate)

    return kept, kept_indices


def deduplicate_exact(
    records: List[dict],
    key_fn: Callable[[dict], tuple],
) -> List[dict]:
    """Remove exact duplicates based on a key function."""
    seen = set()
    unique = []
    for record in records:
        key = key_fn(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique

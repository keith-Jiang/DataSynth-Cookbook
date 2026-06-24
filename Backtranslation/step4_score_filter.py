"""
Step 4: Score backtranslated pairs with LLM-as-judge and filter by quality.
Uses the 5-point rubric from Humpback (Meta, 2023).
"""
import os
import sys
import asyncio
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.io import read_jsonl, write_jsonl
from utils.scoring import score_pairs_async, score_pairs_sync, filter_by_score

from configs import (
    MODEL_NAME, API_KEY, BASE_URL, EXTRA_BODY,
    SCORE_TEMPERATURE, ASYNC_MAX_CONCURRENCY,
    MIN_QUALITY_SCORE,
)


BACKTRANSLATION_RUBRIC = """Rate the quality of the following instruction-response pair as training data on a scale of 1-5:

1 - Very poor: The instruction is nonsensical, too vague to be useful, or the response is completely irrelevant to the instruction
2 - Poor: The instruction is generic/vague and the response is only loosely related, or the response contains mostly noise/boilerplate
3 - Acceptable: The instruction is reasonable and the response is relevant, but the response is incomplete, partially off-topic, or the instruction could be much more specific
4 - Good: The instruction is clear and specific, the response addresses it well with accurate information. Minor room for improvement in either clarity or completeness
5 - Excellent: The instruction is natural and precise (sounds like a real user question), the response is comprehensive, accurate, and well-structured — an ideal training example

Instruction: {instruction}
Response: {response}

Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""


async def score_all_async(pairs: list, batch_size: int = 50) -> list:
    """Score all pairs in batches using async API."""
    all_scores = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        scoring_input = [
            {"instruction": p["instruction"], "response": p["response"]}
            for p in batch
        ]

        scores = await score_pairs_async(
            scoring_input,
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            rubric=BACKTRANSLATION_RUBRIC,
            extra_body=EXTRA_BODY,
            max_concurrency=ASYNC_MAX_CONCURRENCY,
            temperature=SCORE_TEMPERATURE,
        )
        all_scores.extend(scores)
        print(f"  Scored {min(i + batch_size, len(pairs))}/{len(pairs)}")

    return all_scores


def parse_args():
    parser = argparse.ArgumentParser(description="Score and filter backtranslated pairs")
    parser.add_argument("--input_path", type=str, default="output/backtranslated_pairs.jsonl")
    parser.add_argument("--output_path", type=str, default="output/scored_pairs.jsonl")
    parser.add_argument("--filtered_path", type=str, default="output/high_quality_pairs.jsonl")
    parser.add_argument("--min_score", type=int, default=MIN_QUALITY_SCORE)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--sync", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pairs = read_jsonl(args.input_path)
    print(f"Scoring {len(pairs)} backtranslated pairs...")

    if args.sync:
        scoring_input = [
            {"instruction": p["instruction"], "response": p["response"]}
            for p in pairs
        ]
        scores = score_pairs_sync(
            scoring_input,
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            rubric=BACKTRANSLATION_RUBRIC,
            extra_body=EXTRA_BODY,
            temperature=SCORE_TEMPERATURE,
        )
    else:
        scores = asyncio.run(score_all_async(pairs, batch_size=args.batch_size))

    # Attach scores to pairs
    for pair, score_info in zip(pairs, scores):
        pair["quality_score"] = score_info["score"]
        pair["score_reason"] = score_info["reason"]

    # Save all scored pairs
    write_jsonl(args.output_path, pairs)
    print(f"All scored pairs → {args.output_path}")

    # Score distribution
    from collections import Counter
    dist = Counter(p["quality_score"] for p in pairs)
    print(f"Score distribution: {dict(sorted(dist.items()))}")

    # Filter
    high_quality = filter_by_score(pairs, min_score=args.min_score)
    write_jsonl(args.filtered_path, high_quality)
    print(f"High quality (score >= {args.min_score}): {len(high_quality)}/{len(pairs)} → {args.filtered_path}")

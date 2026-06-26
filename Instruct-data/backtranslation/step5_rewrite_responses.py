"""
Step 5 (Optional): Rewrite responses for quality improvement.
Based on Back-and-Forth Translation (Meta, 2024).
For pairs scoring 4, rewrite the response to push quality toward 5.
"""
import os
import sys
import asyncio
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.io import read_jsonl, write_jsonl
from utils.api import make_async_batch_chat_requests, make_batch_chat_requests

from configs import (
    MODEL_NAME, API_KEY, BASE_URL, EXTRA_BODY,
    REWRITE_TEMPERATURE, REWRITE_MAX_TOKENS,
    ASYNC_MAX_CONCURRENCY, REWRITE_SCORE_RANGE,
)


REWRITE_SYSTEM = """You are an expert writer. Given an instruction and its current response, rewrite the response to be:
- Clearer and better organized
- More accurate and comprehensive
- More natural in tone (as if written by a knowledgeable assistant)
- Well-structured with appropriate formatting

Maintain the same core information and topic. Do not add hallucinated facts.
Output ONLY the improved response, nothing else."""

REWRITE_USER = """Instruction: {instruction}

Current Response:
{response}

Improved Response:"""


def build_rewrite_messages(instruction: str, response: str):
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": REWRITE_USER.format(
            instruction=instruction, response=response
        )},
    ]


async def rewrite_async(pairs: list, batch_size: int = 30) -> list:
    """Rewrite responses for selected pairs using async API."""
    results = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        messages_list = [
            build_rewrite_messages(p["instruction"], p["response"])
            for p in batch
        ]

        responses = await make_async_batch_chat_requests(
            messages_list,
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=REWRITE_TEMPERATURE,
            max_tokens=REWRITE_MAX_TOKENS,
            extra_body=EXTRA_BODY,
            max_concurrency=ASYNC_MAX_CONCURRENCY,
        )

        for pair, resp in zip(batch, responses):
            record = dict(pair)
            if resp and resp["content"] and len(resp["content"].split()) > 20:
                record["response"] = resp["content"].strip()
                record["rewritten"] = True
            else:
                record["rewritten"] = False
            results.append(record)

        print(f"  Rewritten {min(i + batch_size, len(pairs))}/{len(pairs)}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite responses for quality improvement")
    parser.add_argument("--input_path", type=str, default="output/high_quality_pairs.jsonl")
    parser.add_argument("--output_path", type=str, default="output/final_backtranslation_data.jsonl")
    parser.add_argument("--rewrite_scores", type=str, default=None,
                        help="Comma-separated scores to rewrite, e.g. '4' or '3,4'. Default from config.")
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--skip_rewrite", action="store_true", help="Skip rewriting, just pass through")
    parser.add_argument("--sync", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pairs = read_jsonl(args.input_path)
    print(f"Loaded {len(pairs)} high-quality pairs")

    if args.skip_rewrite:
        # Just pass through, mark as not rewritten
        for p in pairs:
            p["rewritten"] = False
        write_jsonl(args.output_path, pairs)
        print(f"Skipped rewriting, saved → {args.output_path}")
        sys.exit(0)

    # Determine which scores to rewrite
    if args.rewrite_scores:
        rewrite_scores = set(int(s) for s in args.rewrite_scores.split(","))
    else:
        lo, hi = REWRITE_SCORE_RANGE
        rewrite_scores = set(range(lo, hi + 1))

    to_rewrite = [p for p in pairs if p.get("quality_score") in rewrite_scores]
    passthrough = [p for p in pairs if p.get("quality_score") not in rewrite_scores]
    print(f"Rewriting {len(to_rewrite)} pairs (score in {rewrite_scores}), passing through {len(passthrough)}")

    if to_rewrite:
        if args.sync:
            messages_list = [
                build_rewrite_messages(p["instruction"], p["response"])
                for p in to_rewrite
            ]
            responses = make_batch_chat_requests(
                messages_list,
                model=MODEL_NAME,
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=REWRITE_TEMPERATURE,
                max_tokens=REWRITE_MAX_TOKENS,
                extra_body=EXTRA_BODY,
            )
            rewritten = []
            for pair, resp in zip(to_rewrite, responses):
                record = dict(pair)
                if resp and resp["content"] and len(resp["content"].split()) > 20:
                    record["response"] = resp["content"].strip()
                    record["rewritten"] = True
                else:
                    record["rewritten"] = False
                rewritten.append(record)
        else:
            rewritten = asyncio.run(rewrite_async(to_rewrite, batch_size=args.batch_size))
    else:
        rewritten = []

    # Mark passthrough
    for p in passthrough:
        p["rewritten"] = False

    # Combine
    final = rewritten + passthrough
    write_jsonl(args.output_path, final)

    rewrite_count = sum(1 for r in final if r.get("rewritten"))
    print(f"Done! {rewrite_count} rewritten, {len(final)} total → {args.output_path}")

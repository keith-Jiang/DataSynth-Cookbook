"""
Step 3: Backtranslate — generate instructions from text passages.
For each passage, ask the LLM: "What instruction would produce this text as a response?"
Uses async API for throughput on large corpus.
"""
import os
import sys
import asyncio
import argparse
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.io import read_jsonl, write_jsonl, append_jsonl
from utils.api import make_async_batch_chat_requests, make_batch_chat_requests

from configs import (
    MODEL_NAME, API_KEY, BASE_URL, EXTRA_BODY,
    BACKTRANSLATE_TEMPERATURE, BACKTRANSLATE_MAX_TOKENS,
    ASYNC_MAX_CONCURRENCY,
)


BACKTRANSLATE_SYSTEM = """You are an expert at generating clear, specific instructions. Given a text passage, your job is to write an instruction (a question or task) that, if given to a knowledgeable assistant, would naturally produce the given text as its response.

Requirements:
- The instruction must be self-contained (no references to "the text" or "the passage")
- The instruction should be specific enough that the text is a natural, complete answer
- Do NOT generate instructions that ask to "summarize" or "extract from" a passage
- The instruction should sound like something a real user would ask
- Output ONLY the instruction, nothing else."""

BACKTRANSLATE_USER = """Text:
{passage}

What instruction would produce this text as a response?"""


def build_backtranslate_messages(passage: str):
    """Build messages for backtranslating a single passage."""
    return [
        {"role": "system", "content": BACKTRANSLATE_SYSTEM},
        {"role": "user", "content": BACKTRANSLATE_USER.format(passage=passage)},
    ]


def post_process_instruction(response_text: str) -> str:
    """Clean up the generated instruction."""
    if not response_text:
        return ""
    text = response_text.strip()
    # Remove quotes if wrapped
    if len(text) > 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    # Remove common prefixes
    for prefix in ["Instruction:", "Question:", "Task:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text


async def backtranslate_async(
    passages: list,
    batch_size: int = 50,
) -> list:
    """Backtranslate all passages using async API."""
    results = []

    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        messages_list = [
            build_backtranslate_messages(p["text"])
            for p in batch
        ]

        responses = await make_async_batch_chat_requests(
            messages_list,
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=BACKTRANSLATE_TEMPERATURE,
            max_tokens=BACKTRANSLATE_MAX_TOKENS,
            extra_body=EXTRA_BODY,
            max_concurrency=ASYNC_MAX_CONCURRENCY,
        )

        for passage, resp in zip(batch, responses):
            instruction = ""
            if resp:
                instruction = post_process_instruction(resp["content"])

            results.append({
                "passage_id": passage["passage_id"],
                "source": passage["source"],
                "title": passage["title"],
                "instruction": instruction,
                "response": passage["text"],
            })

        print(f"  Backtranslated {min(i + batch_size, len(passages))}/{len(passages)}")

    return results


def backtranslate_sync(passages: list) -> list:
    """Backtranslate using sync API (fallback for debugging)."""
    results = []
    for passage in tqdm.tqdm(passages, desc="Backtranslating"):
        messages = build_backtranslate_messages(passage["text"])
        from utils.api import make_chat_request
        resp = make_chat_request(
            messages,
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=BACKTRANSLATE_TEMPERATURE,
            max_tokens=BACKTRANSLATE_MAX_TOKENS,
            extra_body=EXTRA_BODY,
        )
        instruction = ""
        if resp:
            instruction = post_process_instruction(resp["content"])

        results.append({
            "passage_id": passage["passage_id"],
            "source": passage["source"],
            "title": passage["title"],
            "instruction": instruction,
            "response": passage["text"],
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Backtranslate: generate instructions from passages")
    parser.add_argument("--input_path", type=str, default="output/passages.jsonl")
    parser.add_argument("--output_path", type=str, default="output/backtranslated_pairs.jsonl")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_passages", type=int, default=None, help="Limit number of passages to process")
    parser.add_argument("--sync", action="store_true", help="Use synchronous API (slower, for debugging)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    passages = read_jsonl(args.input_path)
    if args.max_passages:
        passages = passages[:args.max_passages]
    print(f"Backtranslating {len(passages)} passages...")

    if args.sync:
        results = backtranslate_sync(passages)
    else:
        results = asyncio.run(backtranslate_async(passages, batch_size=args.batch_size))

    # Filter out empty instructions
    valid = [r for r in results if r["instruction"]]
    print(f"Valid backtranslated pairs: {len(valid)}/{len(results)}")

    write_jsonl(args.output_path, valid)
    print(f"Saved → {args.output_path}")

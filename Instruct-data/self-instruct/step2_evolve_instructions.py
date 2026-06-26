import os
import sys
import json
import random
import re
import argparse
import numpy as np
import tqdm
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.dedup import create_rouge_scorer, is_duplicate

from configs import (
    EVOLVE_RATIO,
    EVOLVE_IN_DEPTH_RATIO,
    EVOLVE_BREADTH_COUNT,
    MAX_EVOLVED_INSTRUCTIONS,
    ROUGE_THRESHOLD,
    REQUEST_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
from api_utils import make_batch_chat_requests

random.seed(42)

IN_DEPTH_SYSTEM = """You are an expert at making task instructions more complex and challenging while keeping them clear and actionable.

Rewrite the given instruction to be MORE COMPLEX using ONE of these strategies:
1. Add specific constraints (word limits, format requirements, exclusions, specific data types)
2. Require multi-step reasoning or explicit intermediate steps
3. Add a concrete real-world scenario, persona, or context the task should operate within
4. Decompose the task into sub-tasks or explicitly required sections

The rewritten instruction must:
- Still be a single instruction (not multiple separate tasks)
- Be self-contained and clear
- NOT include examples, solutions, or answers
- Only output the rewritten instruction itself, nothing else."""

IN_BREADTH_SYSTEM = """You are an expert at generating diverse task instructions.

Given an instruction, generate a NEW instruction on a COMPLETELY DIFFERENT topic or domain that requires the SAME type of cognitive skill or output format.

For example:
- "Write a Python function to sort a list" → "Write a SQL query to find duplicate records"
- "Summarize this news article" → "Summarize this meeting transcript"
- "Explain the water cycle" → "Explain how a car engine works"

The new instruction must:
- Be on a clearly different topic/domain from the original
- Require the same type of skill (writing, summarizing, analyzing, comparing, explaining, etc.)
- Be self-contained and clear
- NOT include examples, solutions, or answers
- Only output the new instruction itself, nothing else."""


def post_process_evolved(response_text):
    """Extract the evolved instruction from the response."""
    if response_text is None:
        return None
    text = response_text.strip()
    # Remove common prefixes LLMs sometimes add
    for prefix in ["Complex version:", "New instruction:", "Rewritten:", "Evolved:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Remove quotes if the whole thing is quoted
    if len(text) > 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    if len(text.split()) < 5 or len(text.split()) > 200:
        return None
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--evolve_ratio", type=float, default=EVOLVE_RATIO)
    parser.add_argument("--in_depth_ratio", type=float, default=EVOLVE_IN_DEPTH_RATIO)
    parser.add_argument("--breadth_count", type=int, default=EVOLVE_BREADTH_COUNT)
    parser.add_argument("--max_evolved", type=int, default=MAX_EVOLVED_INSTRUCTIONS)
    parser.add_argument("--request_batch_size", type=int, default=REQUEST_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = os.path.join(args.input_dir, "machine_generated_instructions.jsonl")
    with open(input_path, encoding="utf-8") as fin:
        tasks = [json.loads(line) for line in fin]
    print(f"Loaded {len(tasks)} generated instructions")

    # Select instructions to evolve
    num_to_evolve = min(
        int(len(tasks) * args.evolve_ratio),
        args.max_evolved,
    )
    tasks_to_evolve = random.sample(tasks, num_to_evolve)
    num_in_depth = int(num_to_evolve * args.in_depth_ratio)

    print(f"Evolving {num_to_evolve} instructions: {num_in_depth} in-depth, {num_to_evolve - num_in_depth} in-breadth")

    output_path = os.path.join(args.output_dir, "evolved_instructions.jsonl")

    existing_instructions = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                existing_instructions.add(data["instruction"])
        print(f"Loaded {len(existing_instructions)} existing evolved instructions")

    # Collect all original instructions for ROUGE dedup
    all_original = [t["instruction"] for t in tasks]
    scorer = create_rouge_scorer()

    evolved_count = len(existing_instructions)

    # Phase 1: In-depth evolution
    in_depth_tasks = tasks_to_evolve[:num_in_depth]
    in_depth_batches = [
        in_depth_tasks[i:i + args.request_batch_size]
        for i in range(0, len(in_depth_tasks), args.request_batch_size)
    ]

    with open(output_path, "a", encoding="utf-8") as fout:
        for batch in in_depth_batches:
            batch_messages = []
            for task in batch:
                batch_messages.append([
                    {"role": "system", "content": IN_DEPTH_SYSTEM},
                    {"role": "user", "content": f"Original: {task['instruction']}\nComplex version:"},
                ])

            results = make_batch_chat_requests(
                messages_list=batch_messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )

            for task, result in zip(batch, results):
                if result is None:
                    continue
                evolved = post_process_evolved(result["content"])
                if evolved is None:
                    continue

                # ROUGE dedup
                all_refs = all_original + list(existing_instructions)
                if is_duplicate(evolved, all_refs, scorer, threshold=ROUGE_THRESHOLD):
                    continue

                existing_instructions.add(evolved)
                fout.write(json.dumps({
                    "instruction": evolved,
                    "original_instruction": task["instruction"],
                    "evolve_type": "in-depth",
                }, ensure_ascii=False) + "\n")
                fout.flush()
                evolved_count += 1

        # Phase 2: In-breadth evolution
        breadth_tasks = tasks_to_evolve[num_in_depth:]
        for task in breadth_tasks:
            if evolved_count >= args.max_evolved:
                break
            batch_messages = []
            for _ in range(args.breadth_count):
                batch_messages.append([
                    {"role": "system", "content": IN_BREADTH_SYSTEM},
                    {"role": "user", "content": f"Original: {task['instruction']}\nNew instruction on a different topic:"},
                ])

            results = make_batch_chat_requests(
                messages_list=batch_messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )

            for result in results:
                if evolved_count >= args.max_evolved:
                    break
                if result is None:
                    continue
                evolved = post_process_evolved(result["content"])
                if evolved is None:
                    continue

                all_refs = all_original + list(existing_instructions)
                if is_duplicate(evolved, all_refs, scorer, threshold=ROUGE_THRESHOLD):
                    continue
                    continue

                existing_instructions.add(evolved)
                fout.write(json.dumps({
                    "instruction": evolved,
                    "original_instruction": task["instruction"],
                    "evolve_type": "in-breadth",
                }, ensure_ascii=False) + "\n")
                fout.flush()
                evolved_count += 1

    print(f"\nDone! Generated {evolved_count} evolved instructions → {output_path}")

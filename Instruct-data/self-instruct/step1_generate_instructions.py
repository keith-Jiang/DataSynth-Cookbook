import os
import sys
import json
import random
import re
import string
import argparse
import numpy as np
import tqdm
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.dedup import create_rouge_scorer, compute_rouge_scores, is_duplicate

from configs import (
    NUM_PROMPT_INSTRUCTIONS,
    NUM_INSTRUCTIONS_TO_GENERATE,
    ROUGE_THRESHOLD,
    REQUEST_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
from api_utils import make_batch_chat_requests

random.seed(42)

SYSTEM_PROMPT = "You are a helpful assistant that generates diverse task instructions. Each task should be a short instruction that describes what needs to be done, NOT an example with its answer. Only output the task description itself."

GENERATION_PROMPT_PREFIX = """Come up with a series of tasks. Each task should be a clear, concise instruction describing what needs to be done. Do NOT include examples, solutions, or answers - only the task instruction itself.
"""


def encode_prompt(prompt_instructions):
    prompt = GENERATION_PROMPT_PREFIX
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx + 1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def post_process_response(response_text):
    if response_text is None:
        return []
    raw_instructions = re.split(r"\n\d+\s?\.\s?", response_text)
    if raw_instructions and re.match(r"^\d+\s?\.\s?", raw_instructions[0]):
        raw_instructions[0] = re.sub(r"^\d+\s?\.\s?", "", raw_instructions[0])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        if any(
            find_word_in_string(word, inst)
            for word in [
                "image", "images", "graph", "graphs", "picture", "pictures",
                "file", "files", "map", "maps", "draw", "plot", "go to",
            ]
        ):
            continue
        if inst.startswith("Write a program"):
            continue
        if inst[0] in string.punctuation:
            continue
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_tasks_path", type=str, default="seed_tasks.jsonl")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_instructions_to_generate", type=int, default=NUM_INSTRUCTIONS_TO_GENERATE)
    parser.add_argument("--num_prompt_instructions", type=int, default=NUM_PROMPT_INSTRUCTIONS)
    parser.add_argument("--request_batch_size", type=int, default=REQUEST_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r", encoding="utf-8")]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "machine_generated_instructions.jsonl")

    machine_instructions = []
    request_idx = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} existing machine-generated instructions")

    scorer = create_rouge_scorer()

    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(output_path, "a", encoding="utf-8") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_messages = []
            for _ in range(args.request_batch_size):
                prompt_instructions = random.sample(
                    machine_instructions, min(2, len(machine_instructions))
                )
                prompt_instructions += random.sample(
                    seed_instructions,
                    args.num_prompt_instructions - len(prompt_instructions),
                )
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions)
                batch_messages.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ])

            results = make_batch_chat_requests(
                messages_list=batch_messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )

            instructions = []
            all_metadata = []
            for result in results:
                if result is None:
                    continue
                new_instructions = post_process_response(result["content"])
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)

            for inst, metadata in zip(instructions, all_metadata):
                all_refs = seed_instructions + machine_instructions
                rouge_scores = compute_rouge_scores(inst, all_refs, scorer)
                if not rouge_scores or max(rouge_scores) > ROUGE_THRESHOLD:
                    continue
                most_similar_instructions = {
                    all_refs[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                machine_instructions.append(inst)
                fout.write(
                    json.dumps({
                        "instruction": inst,
                        "most_similar": most_similar_instructions,
                        "avg_similarity_score": float(np.mean(rouge_scores)),
                        "metadata": metadata,
                        "request_idx": request_idx,
                    }) + "\n"
                )
                fout.flush()
                progress_bar.update(1)
            request_idx += 1

    print(f"\nDone! Generated {len(machine_instructions)} instructions.")

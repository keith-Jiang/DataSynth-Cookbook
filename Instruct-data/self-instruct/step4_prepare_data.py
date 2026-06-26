import os
import sys
import json
import re
import random
import argparse
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.filtering import filter_invalid_instances, filter_duplicate_instances
from utils.dedup import deduplicate_exact

random.seed(123)


def parse_input_output(response_text):
    if re.findall(r"Output\s*\d*\s*:", response_text):
        inst_input = re.split(r"Output\s*\d*\s*:", response_text)[0].strip()
        inst_output = re.split(r"Output\s*\d*\s*:", response_text)[1].strip()
    else:
        inst_input = ""
        inst_output = response_text.strip()
    if re.findall(r"Input\s*\d*\s*:", inst_output):
        inst_output = re.split(r"Input\s*\d*\s*:", inst_output)[0].strip()
    inst_input = re.sub(r"^Input\s*\d*\s*:", "", inst_input).strip()
    return inst_input, inst_output


def parse_instances_for_generation_task(raw_text, instruction):
    instances = []
    raw_text = raw_text.strip()
    if re.findall(r"Example\s?\d*\.?", raw_text):
        instance_texts = re.split(r"Example\s?\d*\.?", raw_text)
        instance_texts = [it.strip() for it in instance_texts if it.strip() != ""]
        for instance_text in instance_texts:
            inst_input, inst_output = parse_input_output(instance_text)
            instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    elif re.findall(r"Output\s*\d*\s*:", raw_text):
        inst_input, inst_output = parse_input_output(raw_text)
        instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    else:
        return []
    instances = filter_invalid_instances(instances)
    instances = filter_duplicate_instances(instances)
    return instances


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed_tasks_path", type=str, default="seed_tasks.jsonl")
    parser.add_argument("--include_seed_tasks", action="store_true")
    parser.add_argument("--num_instructions", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    instances_path = os.path.join(args.input_dir, "machine_generated_instances.jsonl")
    with open(instances_path, encoding="utf-8") as fin:
        generated_tasks = [json.loads(line) for line in fin]
    if args.num_instructions is not None:
        generated_tasks = generated_tasks[:args.num_instructions]
    print(f"Loaded {len(generated_tasks)} generated tasks")

    training_instances = []
    for task in tqdm.tqdm(generated_tasks):
        instruction = task["instruction"]
        persona_role = task.get("persona_role", "")
        raw_text = task.get("raw_instances", "")
        if not raw_text:
            continue

        task_instances = parse_instances_for_generation_task(raw_text, instruction)
        task_instances = random.sample(task_instances, min(len(task_instances), 5))

        for inst in task_instances:
            training_instances.append({
                "instruction": inst[0],
                "input": inst[1],
                "output": inst[2],
                "persona": persona_role,
            })

    if args.include_seed_tasks:
        seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r", encoding="utf-8")]
        for task in seed_tasks:
            for instance in task["instances"]:
                training_instances.append({
                    "instruction": task["instruction"],
                    "input": instance["input"],
                    "output": instance["output"],
                    "persona": "",
                })
        print(f"Included {len(seed_tasks)} seed tasks")

    # Deduplicate
    training_instances = deduplicate_exact(
        training_instances,
        key_fn=lambda x: (x["instruction"], x["input"], x["output"]),
    )

    random.shuffle(training_instances)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "final_training_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as fout:
        for instance in training_instances:
            fout.write(json.dumps(instance, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(training_instances)} training instances to {output_path}")
    unique_instructions = set(it["instruction"] for it in training_instances)
    print(f"Unique instructions: {len(unique_instructions)}")

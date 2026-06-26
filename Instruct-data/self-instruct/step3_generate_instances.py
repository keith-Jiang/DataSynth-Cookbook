import os
import json
import random
import argparse
import tqdm

from api_utils import make_batch_chat_requests
from configs import REQUEST_BATCH_SIZE, PERSONA_TEMPERATURE, MAX_INSTANCES_PER_TASK
from personas import PERSONAS

random.seed(42)

INSTANCE_GENERATION_TEMPLATE = """Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: Which exercises are best for reducing belly fat at home?
Output:
- Lying Leg Raises
- Leg In And Out
- Plank
- Side Plank
- Sit-ups

Task: Extract all the country names in the paragraph, list them separated by commas.
Example 1
Paragraph: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958.
Output: English, British, Jamaica, the United Kingdom

Task: Converting 85 F to Celsius.
Output: 85°F = 29.44°C

Task: Sort the given list ascendingly.
Example 1
List: [10, 92, 2, 5, -4, 92, 5, 101]
Output: [-4, 2, 5, 5, 10, 92, 92, 101]
Example 2
List: [9.99, 10, -5, -1000, 5e6, 999]
Output: [-1000, -5, 9.99, 10, 999, 5e6]

Task: Suggest a better and more professional rephrasing of the following sentence.
Example 1
Sentence: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it.
Output: This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it.

Task: Select the oldest person from the given list.
Example 1
List: George Washington, Confucius, Michael Jordan, Michelangelo
Output: Confucius

Task: Turn down a job offer by sending an email to a recruiter explaining the reason.
Output: Hi [Recruiter],
Thank you so much for the generous offer to join your team. As we discussed, I've admired the company for a number of years. However, after further consideration of where I currently am in my career, I've decided to accept an offer at another company.
I would love to stay in touch and have started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.
Thanks again,
[Your Name]

Task:"""


def make_persona_system_prompt(persona):
    return (
        f"You are {persona['role']}. "
        f"Your communication style: {persona['style']}. "
        f"Generate diverse, realistic examples for the given task. "
        f"Provide input-output pairs that reflect your role's perspective and communication style. "
        f"Generate examples ONLY for the last task listed. Do NOT repeat examples from earlier tasks."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--include_evolved", action="store_true", help="Also generate instances for evolved instructions")
    parser.add_argument("--num_instructions", type=int, default=None)
    parser.add_argument("--request_batch_size", type=int, default=REQUEST_BATCH_SIZE)
    parser.add_argument("--max_instances", type=int, default=MAX_INSTANCES_PER_TASK)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    instructions_path = os.path.join(args.input_dir, "machine_generated_instructions.jsonl")
    with open(instructions_path, encoding="utf-8") as fin:
        tasks = [json.loads(line) for line in fin]

    if args.include_evolved:
        evolved_path = os.path.join(args.input_dir, "evolved_instructions.jsonl")
        if os.path.exists(evolved_path):
            with open(evolved_path, encoding="utf-8") as fin:
                evolved_tasks = [json.loads(line) for line in fin]
            tasks += evolved_tasks
            print(f"Loaded {len(evolved_tasks)} evolved instructions")

    if args.num_instructions is not None:
        tasks = tasks[:args.num_instructions]

    print(f"Generating instances for {len(tasks)} instructions with persona-driven responses")

    output_path = os.path.join(args.output_dir, "machine_generated_instances.jsonl")

    existing = {}
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as fin:
            for line in fin:
                try:
                    data = json.loads(line)
                    existing[data["instruction"]] = data
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(existing)} existing instances")

    progress_bar = tqdm.tqdm(total=len(tasks))

    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx:batch_idx + args.request_batch_size]

            if all(d["instruction"] in existing for d in batch):
                for d in batch:
                    fout.write(json.dumps(existing[d["instruction"]], ensure_ascii=False) + "\n")
                progress_bar.update(len(batch))
                continue

            batch_messages = []
            batch_personas = []
            for task in batch:
                persona = random.choice(PERSONAS)
                batch_personas.append(persona)
                prompt = INSTANCE_GENERATION_TEMPLATE + " " + task["instruction"].strip() + "\n"
                batch_messages.append([
                    {"role": "system", "content": make_persona_system_prompt(persona)},
                    {"role": "user", "content": prompt},
                ])

            results = make_batch_chat_requests(
                messages_list=batch_messages,
                temperature=PERSONA_TEMPERATURE,
                max_tokens=350,
            )

            for task, result, persona in zip(batch, results, batch_personas):
                data = {
                    "instruction": task["instruction"],
                    "persona_role": persona["role"],
                    "persona_style": persona["style"],
                    "raw_instances": result["content"] if result else "",
                    "instance_metadata": result if result else None,
                }
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()
            progress_bar.update(len(batch))

    print("Done generating instances.")

import argparse
import subprocess
import sys


def run_step(script, args_list):
    cmd = [sys.executable, script] + args_list
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=None)
    if result.returncode != 0:
        print(f"Step failed with return code {result.returncode}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Instruct-data pipeline")
    parser.add_argument(
        "--num_instructions",
        type=int,
        default=100,
        help="Number of instructions to generate in Step 1",
    )
    parser.add_argument(
        "--evolve_ratio",
        type=float,
        default=0.5,
        help="Fraction of generated instructions to evolve in Step 2",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        default="seed_tasks.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
    )
    parser.add_argument(
        "--skip_to",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Skip to a specific step (1-4)",
    )
    parser.add_argument(
        "--include_seed_tasks",
        action="store_true",
        help="Include seed tasks in final training data",
    )
    parser.add_argument(
        "--include_evolved",
        action="store_true",
        help="Include evolved instructions in Step 3 instance generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.skip_to <= 1:
        run_step("step1_generate_instructions.py", [
            "--seed_tasks_path", args.seed_tasks_path,
            "--output_dir", args.output_dir,
            "--num_instructions_to_generate", str(args.num_instructions),
        ])

    if args.skip_to <= 2:
        run_step("step2_evolve_instructions.py", [
            "--input_dir", args.output_dir,
            "--output_dir", args.output_dir,
            "--evolve_ratio", str(args.evolve_ratio),
        ])

    if args.skip_to <= 3:
        step3_args = [
            "--input_dir", args.output_dir,
            "--output_dir", args.output_dir,
        ]
        if args.include_evolved:
            step3_args.append("--include_evolved")
        run_step("step3_generate_instances.py", step3_args)

    if args.skip_to <= 4:
        step4_args = [
            "--input_dir", args.output_dir,
            "--output_dir", args.output_dir,
            "--seed_tasks_path", args.seed_tasks_path,
        ]
        if args.include_seed_tasks:
            step4_args.append("--include_seed_tasks")
        run_step("step4_prepare_data.py", step4_args)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Results saved to: {args.output_dir}/final_training_data.jsonl")
    print(f"{'='*60}")

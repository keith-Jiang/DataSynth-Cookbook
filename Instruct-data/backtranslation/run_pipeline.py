"""
Backtranslation Pipeline Runner.
Orchestrates: corpus collection → segmentation → backtranslation → scoring → rewriting.
"""
import argparse
import subprocess
import sys


def run_step(script, args_list):
    cmd = [sys.executable, script] + args_list
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Step failed with return code {result.returncode}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Backtranslation pipeline")
    parser.add_argument("--max_wiki", type=int, default=50, help="Max Wikipedia articles to fetch")
    parser.add_argument("--max_stack", type=int, default=30, help="Max StackOverflow posts to fetch")
    parser.add_argument("--max_passages", type=int, default=None, help="Limit passages for backtranslation")
    parser.add_argument("--min_score", type=int, default=4, help="Min quality score to keep")
    parser.add_argument("--skip_rewrite", action="store_true", help="Skip response rewriting step")
    parser.add_argument("--sync", action="store_true", help="Use sync API (slower, for debugging)")
    parser.add_argument("--skip_to", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Skip to a specific step")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Collect corpus
    if args.skip_to <= 1:
        step1_args = [
            "--max_wiki", str(args.max_wiki),
            "--max_stack", str(args.max_stack),
        ]
        run_step("step1_collect_corpus.py", step1_args)

    # Step 2: Clean and segment
    if args.skip_to <= 2:
        run_step("step2_clean_segment.py", [])

    # Step 3: Backtranslate
    if args.skip_to <= 3:
        step3_args = []
        if args.max_passages:
            step3_args += ["--max_passages", str(args.max_passages)]
        if args.sync:
            step3_args.append("--sync")
        run_step("step3_backtranslate.py", step3_args)

    # Step 4: Score and filter
    if args.skip_to <= 4:
        step4_args = ["--min_score", str(args.min_score)]
        if args.sync:
            step4_args.append("--sync")
        run_step("step4_score_filter.py", step4_args)

    # Step 5: Rewrite responses
    if args.skip_to <= 5:
        step5_args = []
        if args.skip_rewrite:
            step5_args.append("--skip_rewrite")
        if args.sync:
            step5_args.append("--sync")
        run_step("step5_rewrite_responses.py", step5_args)

    print(f"\n{'='*60}")
    print("Backtranslation pipeline complete!")
    print("Final output: output/final_backtranslation_data.jsonl")
    print(f"{'='*60}")

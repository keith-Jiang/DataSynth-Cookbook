"""
Merge datasets from all pipelines into a unified training file.
Performs cross-pipeline dedup and optional quality filtering.
"""
import os
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from utils.io import read_jsonl, write_jsonl
from utils.dedup import deduplicate_exact, deduplicate_by_rouge

random.seed(42)


def normalize_instruct_data(record: dict) -> dict:
    """Normalize Instruct-data format to unified schema."""
    return {
        "instruction": record["instruction"],
        "input": record.get("input", ""),
        "output": record.get("output", ""),
        "source": "instruct-data",
        "metadata": {
            "persona": record.get("persona", ""),
        },
    }


def normalize_backtranslation(record: dict) -> dict:
    """Normalize Backtranslation format to unified schema."""
    return {
        "instruction": record["instruction"],
        "input": "",
        "output": record.get("response", ""),
        "source": "backtranslation",
        "metadata": {
            "original_source": record.get("source", ""),
            "quality_score": record.get("quality_score", 0),
            "rewritten": record.get("rewritten", False),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Merge datasets from all pipelines")
    parser.add_argument(
        "--instruct_path", type=str,
        default="Instruct-data/self-instruct/output/final_training_data.jsonl",
    )
    parser.add_argument(
        "--backtranslation_path", type=str,
        default="Instruct-data/backtranslation/output/final_backtranslation_data.jsonl",
    )
    parser.add_argument("--output_path", type=str, default="merged_training_data.jsonl")
    parser.add_argument("--rouge_dedup", action="store_true",
                        help="Cross-pipeline ROUGE dedup on instruction field (slower)")
    parser.add_argument("--rouge_threshold", type=float, default=0.7)
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip missing pipeline outputs instead of erroring")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_records = []

    # Load Instruct-data
    if os.path.exists(args.instruct_path):
        raw = read_jsonl(args.instruct_path)
        instruct_records = [normalize_instruct_data(r) for r in raw]
        all_records.extend(instruct_records)
        print(f"[Instruct-data] Loaded {len(instruct_records)} records")
    elif not args.skip_missing:
        print(f"Error: {args.instruct_path} not found. Use --skip_missing to ignore.")
        sys.exit(1)
    else:
        print(f"[Instruct-data] Skipped (file not found)")

    # Load Backtranslation
    if os.path.exists(args.backtranslation_path):
        raw = read_jsonl(args.backtranslation_path)
        bt_records = [normalize_backtranslation(r) for r in raw]
        all_records.extend(bt_records)
        print(f"[Backtranslation] Loaded {len(bt_records)} records")
    elif not args.skip_missing:
        print(f"Error: {args.backtranslation_path} not found. Use --skip_missing to ignore.")
        sys.exit(1)
    else:
        print(f"[Backtranslation] Skipped (file not found)")

    if not all_records:
        print("No records to merge.")
        sys.exit(0)

    # Exact dedup by (instruction, output)
    before = len(all_records)
    all_records = deduplicate_exact(
        all_records,
        key_fn=lambda r: (r["instruction"].strip().lower(), r["output"].strip()[:200]),
    )
    print(f"[Exact dedup] {before} → {len(all_records)} (removed {before - len(all_records)})")

    # Optional ROUGE dedup on instructions
    if args.rouge_dedup:
        print(f"[ROUGE dedup] Running with threshold={args.rouge_threshold}...")
        instructions = [r["instruction"] for r in all_records]
        kept, kept_indices = deduplicate_by_rouge(
            instructions, threshold=args.rouge_threshold
        )
        all_records = [all_records[i] for i in kept_indices]
        print(f"[ROUGE dedup] Kept {len(all_records)} records")

    # Shuffle
    random.shuffle(all_records)

    # Write
    write_jsonl(args.output_path, all_records)

    # Stats
    from collections import Counter
    source_dist = Counter(r["source"] for r in all_records)
    print(f"\nMerged {len(all_records)} records → {args.output_path}")
    print(f"Source distribution: {dict(source_dist)}")

"""Summarize an EvoEval eval_results.json (one greedy sample per problem) for Harbor parity."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation-seconds", type=float, default=None)
    parser.add_argument("--total-seconds", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    results = json.loads(args.eval_results.read_text())["eval"]
    per_task = {}
    for task_id, samples in sorted(results.items(), key=lambda kv: int(kv[0].split("/")[1])):
        assert len(samples) == 1, f"{task_id}: expected one greedy sample"
        per_task[task_id] = {
            "base_status": samples[0]["base_status"],
            "reward": 1.0 if samples[0]["base_status"] == "pass" else 0.0,
        }
    n = len(per_task)
    n_pass = sum(v["reward"] for v in per_task.values())
    summary = {
        "side": "original",
        "dataset": args.dataset,
        "model": args.model,
        "n_tasks": n,
        "n_pass": int(n_pass),
        "pass@1": n_pass / n if n else 0.0,
        "generation_seconds": args.generation_seconds,
        "total_seconds": args.total_seconds,
        "per_task": per_task,
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"{args.dataset}: pass@1 = {summary['pass@1']:.3f} ({int(n_pass)}/{n})")


if __name__ == "__main__":
    main()

"""Summarize one original-side run scored by the unmodified ``evoeval.evaluate``.

Reads ``<run-dir>/samples_eval_results.json`` (written by
``evoeval.evaluate --samples <run-dir>/samples.jsonl``) and reports pass@1 over
the problems selected for the run (``selected_task_ids.json``; all 100 for a
full run). With one sample per problem, upstream pass@1 is the fraction of
problems whose ``base_status`` is ``pass``.

    python harbor_parity/summarize.py <run-dir> [--out result.json]
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    selected = json.loads((run / "selected_task_ids.json").read_text())
    metas = {}
    incomplete = []
    for tid in selected:
        meta_p = run / "tasks" / tid.replace("/", "_") / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        if not meta.get("completed"):
            incomplete.append(tid)
        metas[tid] = meta
    if incomplete:
        raise SystemExit(
            f"refusing to summarize: {len(incomplete)} selected attempt(s) have no completed "
            f"run (infrastructure errors; re-run run_agent.py --resume): {incomplete}"
        )
    samples = {}
    for line in (run / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        samples[row["task_id"]] = row["solution"]
    results = json.loads((run / "samples_eval_results.json").read_text())["eval"]
    stale = [tid for tid in selected if results[tid][0]["solution"] != samples[tid]]
    if stale:
        raise SystemExit(
            f"refusing to summarize: samples_eval_results.json does not match samples.jsonl "
            f"for {stale}; delete it and re-run evoeval.evaluate"
        )
    per_task = {}
    for tid in selected:
        meta = metas[tid]
        per_task[tid] = {
            "status": results[tid][0]["base_status"],
            "reward": 1 if results[tid][0]["base_status"] == "pass" else 0,
            "solution_found": meta.get("solution_found"),
            "agent_exit_code": meta.get("agent_exit_code"),
            "agent_timed_out": meta.get("agent_timed_out"),
            "agent_seconds": meta.get("agent_seconds"),
        }
    n_pass = sum(v["reward"] for v in per_task.values())
    summary = {
        "side": "original",
        "run_dir": str(run),
        "metric": "pass@1",
        "n_tasks": len(selected),
        "n_pass": n_pass,
        "pass@1": round(100.0 * n_pass / len(selected), 2),
        "per_task": per_task,
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Original-side parity run for the Harbor EvoEval adapter.
#
# Runs the original EvoEval pipeline unchanged in its logic:
#   1. codegen/generate.py   (greedy, one sample per problem, OpenAIChatDecoder)
#   2. tool/sanitize.py --eof
#   3. evoeval/evaluate.py   (pass@1)
# and writes a per-task summary with harbor_parity/summarize.py.
#
# Usage:
#   harbor_parity/run_original_parity.sh <run_dir> [n_parallel_generation]
#
# Environment:
#   OPENAI_API_KEY, OPENAI_BASE_URL   OpenAI-compatible endpoint that serves MODEL
#   MODEL                    default deepseek-v4.1-flash (must have an entry in codegen/model.py)
#   DATASET                  default EvoEval_difficult
#   EVOEVAL_MAX_NEW_TOKENS   default 32768 (reasoning model; original default is 512)
#   EVOEVAL_REQUEST_TIMEOUT  default 900 seconds per request (original is 100)
#   GEN_PY                   python used for generation/sanitize (default: python)
#   EVAL_PY                  python used for evaluation (default: python; Python 3.9 in parity)
#   EVAL_PARALLEL            evaluate.py --parallel (default 8)
#   TASK_IDS                 optional space-separated problem numbers (e.g. "3 17 42") for a
#                            sanity run; uses EvoEval's EVOEVAL_OVERRIDE_PATH with a subset file
#   EVOEVAL_PROMPT_LF        default 1: send prompts with LF line endings, as Harbor agents get them
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -e "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
  # evaluate.py would load an existing eval_results.json instead of grading new samples.
  echo "error: run directory $1 already exists and is not empty; use a new one" >&2
  exit 1
fi
RUN_DIR="$(mkdir -p "$1" && cd "$1" && pwd)"
NPAR="${2:-10}"
MODEL="${MODEL:-deepseek-v4.1-flash}"
DATASET="${DATASET:-EvoEval_difficult}"
GEN_PY="${GEN_PY:-python}"
EVAL_PY="${EVAL_PY:-python}"
EVAL_PARALLEL="${EVAL_PARALLEL:-8}"
export EVOEVAL_MAX_NEW_TOKENS="${EVOEVAL_MAX_NEW_TOKENS:-32768}"
export EVOEVAL_REQUEST_TIMEOUT="${EVOEVAL_REQUEST_TIMEOUT:-900}"
export EVOEVAL_PROMPT_LF="${EVOEVAL_PROMPT_LF:-1}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${TASK_IDS:-}" ]; then
  mkdir -p "$RUN_DIR/data"
  "$GEN_PY" - "$DATASET" "$RUN_DIR/data/$DATASET.jsonl" $TASK_IDS <<'PY'
import json, sys
from evoeval.data import get_evo_eval
dataset, out, ids = sys.argv[1], sys.argv[2], {f"EvoEval/{int(i)}" for i in sys.argv[3:]}
with open(out, "w") as f:
    for task_id, problem in get_evo_eval(dataset).items():
        if task_id in ids:
            f.write(json.dumps(problem) + "\n")
PY
  export EVOEVAL_OVERRIDE_PATH="$RUN_DIR/data"
fi

# Problem ids are "EvoEval/<n>"; generate.py --id-range takes [low, high).
# Split the sorted ids into NPAR contiguous chunks, one generate.py process each.
mapfile -t RANGES < <("$GEN_PY" -c "
from evoeval.data import get_evo_eval
ids = sorted(int(t.split('/')[1]) for t in get_evo_eval('$DATASET'))
n = max(1, min($NPAR, len(ids))); k = -(-len(ids) // n)
for c in [ids[i:i + k] for i in range(0, len(ids), k)]:
    print(c[0], c[-1] + 1)
")

START=$(date +%s)
pids=()
for r in "${RANGES[@]}"; do
  read -r low high <<< "$r"
  (cd "$REPO/codegen" && "$GEN_PY" generate.py \
      --model "$MODEL" --bs 1 --temperature 0 --n_samples 1 --greedy \
      --dataset "$DATASET" --root "$RUN_DIR" --id-range "$low" "$high" \
      > "$RUN_DIR/generate_${low}_${high}.log" 2>&1) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
GEN_END=$(date +%s)

WORKDIR="$RUN_DIR/$DATASET/$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')_temp_0.0"
(cd "$REPO" && "$GEN_PY" tool/sanitize.py --folder "$WORKDIR" --dataset "$DATASET" --eof \
    > "$RUN_DIR/sanitize.log" 2>&1)

(cd "$REPO" && "$EVAL_PY" -m evoeval.evaluate --dataset "$DATASET" \
    --samples "$WORKDIR-sanitized" --parallel "$EVAL_PARALLEL" \
    > "$RUN_DIR/evaluate.log" 2>&1)
END=$(date +%s)

"$GEN_PY" "$REPO/harbor_parity/summarize.py" \
    --eval-results "$WORKDIR-sanitized/eval_results.json" \
    --dataset "$DATASET" --model "$MODEL" \
    --generation-seconds $(( GEN_END - START )) --total-seconds $(( END - START )) \
    --output "$RUN_DIR/summary.json"
tail -n 3 "$RUN_DIR/evaluate.log"

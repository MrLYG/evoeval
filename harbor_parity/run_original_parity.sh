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
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(mkdir -p "$1" && cd "$1" && pwd)"
NPAR="${2:-10}"
MODEL="${MODEL:-deepseek-v4.1-flash}"
DATASET="${DATASET:-EvoEval_difficult}"
GEN_PY="${GEN_PY:-python}"
EVAL_PY="${EVAL_PY:-python}"
EVAL_PARALLEL="${EVAL_PARALLEL:-8}"
export EVOEVAL_MAX_NEW_TOKENS="${EVOEVAL_MAX_NEW_TOKENS:-32768}"
export EVOEVAL_REQUEST_TIMEOUT="${EVOEVAL_REQUEST_TIMEOUT:-900}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Problem ids are "EvoEval/<n>"; generate.py --id-range takes [low, high).
N_PROBLEMS="$("$GEN_PY" -c "from evoeval.data import get_evo_eval; print(len(get_evo_eval('$DATASET')))")"
CHUNK=$(( (N_PROBLEMS + NPAR - 1) / NPAR ))

START=$(date +%s)
pids=()
for ((low = 0; low < N_PROBLEMS; low += CHUNK)); do
  high=$(( low + CHUNK ))
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

# EvoEval: original-side parity runner for the Harbor adapter

This directory reproduces the **original-benchmark side** of the parity experiment for the Harbor adapter `evoeval-auto`. The adapter lives at [harbor-framework/adapters](https://github.com/harbor-framework/adapters), in `src/evoeval-auto`.

EvoEval's own harness asks a model once and scores the extracted function. Harbor runs agents, so both sides run the **same CLI agent**: `mini-swe-agent==2.4.6` with `openai/deepseek-v4.1-flash`. Each problem's answer is then scored with EvoEval's `evoeval.evaluate`, which here has one documented bugfix (see [Changes from upstream](#changes-from-upstream)). All other code in this fork is upstream EvoEval, unchanged.

## What `run_agent.py` does

For each problem of the dataset (default `EvoEval_difficult`, all 100), the runner does four things:

1. **Renders the instruction.** It uses `instruction_template.md`, which is byte-identical to the Harbor task's `instruction.md`. The text is the EvoEval API-model prompt `Please complete the following code snippet.` plus one sentence saying to save the code to `/workspace/solution.py`.
2. **Creates a Daytona sandbox.** It is built from `Dockerfile`, which is identical to the Harbor task environment: `python:3.10.22-slim`, `numpy==2.2.6`, curl, git and build-essential. The sandbox has 2 CPUs, 4 GB of RAM and 10 GB of disk.
3. **Installs and runs mini-swe-agent the same way Harbor 0.23.0 does.** Everything matches:
   - the install command: uv, Python 3.12, then `uv tool install mini-swe-agent==2.4.6 --with litellm --with orjson --with fastapi`
   - the command line: `mini-swe-agent --yolo --model=... --task=<instruction> --output=... --cost-limit 0 -c mini -c model.model_kwargs.timeout=3600 -c model.model_kwargs.extra_headers.X-Session-ID=... --exit-immediately`
   - the environment variables: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `MSWEA_CONFIGURED`, `MSWEA_COST_TRACKING`
   - the working directory, `/workspace`
   - the agent timeout, 1800 s
   - the network policy: like the Harbor task (`[agent] network_mode = "no-network"`), the agent has no network while it runs, except the hosts given with `--allow-agent-host` (its model API). Harbor and this runner apply it the same way, with Daytona's domain allowlist after the agent install. The install step keeps network access. One difference: Harbor 0.23.0 restores the public network after the agent phase and then collects `/workspace/solution.py`, while this runner keeps the restriction until the sandbox is deleted. Files are downloaded through the Daytona API, so they do not need the sandbox's network. The order matters only for an agent that leaves a background process running to change its answer after its run. The EvoEval data, including canonical solutions and test inputs, is public, so an agent with open network could download it.
4. **Collects the result.** It downloads `/workspace/solution.py` and the agent logs, then deletes the sandbox.

It writes `samples.jsonl` with one `{"task_id", "solution"}` line per problem. The `solution` is the file content as written, or an empty string if the agent wrote nothing.

Infrastructure errors are not scored and stay pending until a `--resume` run completes them. These are: a failed sandbox start, a failed agent install, a failed network-policy change, and a failed check or download of `/workspace/solution.py`. A file that is missing after the agent finished is the agent's result, and it is scored as an empty solution.

When the agent timeout (1800 s) expires, the runner deletes the command's Daytona session. This kills the agent before `/workspace/solution.py` is collected. A run directory can be resumed only with `--resume`, identical settings and the same task selection; otherwise use a new `--run-dir`.

Whenever `samples.jsonl` is rewritten, the runner deletes `samples_eval_results.json`. `evoeval.evaluate` would otherwise reuse the old results. `summarize.py` refuses to report a score in two cases: when any selected problem has no completed attempt, and when the evaluated solutions differ from `samples.jsonl`.

## Reproduce

```bash
git clone -b harbor-parity https://github.com/<fork-owner>/evoeval && cd evoeval
uv venv --python 3.10.22 .venv
uv pip install --python .venv/bin/python -e . -r harbor_parity/requirements.txt

# keys: OPENAI_API_KEY, OPENAI_BASE_URL (OpenAI-compatible endpoint), DAYTONA_API_KEY
RUN=harbor_parity/runs/run1
.venv/bin/python harbor_parity/run_agent.py --run-dir $RUN --env-file /path/to/keys.env \
    --model openai/deepseek-v4.1-flash --agent-version 2.4.6 --n-concurrent 6 \
    --agent-env UV_EXCLUDE_NEWER=2026-10-17T00:00:00Z \
    --allow-agent-host <host of OPENAI_BASE_URL>

# score with this fork's evoeval.evaluate (upstream plus the float fix; all 100 problems)
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m evoeval.evaluate \
    --dataset EvoEval_difficult --samples $RUN/samples.jsonl --parallel 8
.venv/bin/python harbor_parity/summarize.py $RUN --out $RUN/result.json
```

For a sanity check on a few problems, pass `--task-ids EvoEval/0 EvoEval/1 ...`. `samples.jsonl` still lists all 100 problems, because `evoeval.evaluate` requires every problem; the unselected ones get empty solutions. `summarize.py` reports pass@1 over the selected problems only.

## Parity results

Pending: the parity runs with the agent-phase network policy.

## Agent install pin

Harbor's mini-swe-agent installer leaves `litellm` unpinned. Since 2026-10-18 this resolves to the litellm 1.106 line, which fails every tool call with `No module named 'mcp'`. Both sides therefore set `UV_EXCLUDE_NEWER=2026-10-17T00:00:00Z` for the agent's setup and run commands:

- **Harbor side:** `--ae UV_EXCLUDE_NEWER=...`
- **This runner:** `--agent-env UV_EXCLUDE_NEWER=...`

With this setting uv resolves `litellm==1.105.1`. The runner records the installed version in each task's `meta.json`.

## Evaluation settings

- **`OPENBLAS_NUM_THREADS=1`.** The Harbor verifier sets the same value. The upstream evaluator caps each candidate process at 4 GiB of virtual memory, and numpy's thread pool grows with the CPU count. On many-core hosts the pool alone can push a correct solution over the cap.
- **`--parallel 8`.** This is an upstream flag. Its default is `cpu_count // 2` worker processes, which on a 112-core host would put a heavy load on a shared machine while time limits are being measured.
- **Dependencies.** `evalplus==0.2.0`, the release current when EvoEval 0.1.0 was published, and `numpy==2.2.6`. A newer evalplus imports heavy libraries that count against the 4 GiB cap.
- **No sanitizer.** The upstream `tool/sanitize.py` extracts a function from a raw chat reply. Agents write files, so the file is scored as written. The Harbor side does the same.

## Changes from upstream

The base is upstream `evo-eval/evoeval` at `d5ca310`. Its evaluation code is identical to the `evoeval==0.1.0` release. This branch adds the `harbor_parity/` directory, a short note at the top of `README.md`, and one documented scoring bugfix in `evoeval/eval_test/__init__.py`: float values must have the same number of elements. The exact patch is in `UPSTREAM_CHANGES.md`, and the Harbor adapter applies the same fix. Data loading, prompts, the rest of `evoeval.evaluate` and the oracles are unchanged:

```
$ git diff --name-status d5ca310 HEAD
M       README.md
M       evoeval/eval_test/__init__.py
A       harbor_parity/.gitignore
A       harbor_parity/Dockerfile
A       harbor_parity/README.md
A       harbor_parity/UPSTREAM_CHANGES.md
A       harbor_parity/instruction_template.md
A       harbor_parity/requirements.txt
A       harbor_parity/run_agent.py
A       harbor_parity/summarize.py
```

Check it with `git diff d5ca310 HEAD -- . ':!harbor_parity'`; the output shows only the `README.md` note and the `is_equal` fix.

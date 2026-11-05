# <img src="resources/butterfly_dark.png" width="32px" height="auto"> EvoEval: Evolving Coding Benchmarks via LLM

<p align="center">
    <a href="https://evo-eval.github.io/leaderboard.html"><img src="https://img.shields.io/badge/🏆-LeaderBoard-8e7cc3?style=for-the-badge"></a>
    <a href="https://evo-eval.github.io/visualization.html"><img src="https://img.shields.io/badge/🔮-Visualization-3d85c6?style=for-the-badge"></a>
    <a href="https://arxiv.org/abs/2403.19114"><img src="https://img.shields.io/badge/📃-Arxiv-b31b1b?style=for-the-badge"></a>
    <a href="https://huggingface.co/evoeval/"><img src="https://img.shields.io/badge/🤗-Huggingface-f59e0b?style=for-the-badge"></a>
    <a href="https://pypi.org/project/evoeval/"><img src="https://img.shields.io/badge/0.1.0-Pypi-3b719f?style=for-the-badge&logo=pypi"></a>
</p>

<p align="center">
    <big><a href="#-quick-start">⚡Quick Start</a></big> |
    <big><a href="#-benchmarks">🔠Benchmarks</a></big> |
    <big><a href="#-llm-generated-code">🤖LLM Generated Code</a></big> |
    <big><a href="#-citation">📝Citation</a></big> |
    <big><a href="#-acknowledgement">🙏Acknowledgement</a></big>
</p>

## <img src="resources/butterfly_dark.png" width="23px" height="auto"> About 

**EvoEval**<sup>1</sup> is a holistic benchmark suite created by _evolving_ **HumanEval** problems:
- 🔥 Contains **828** new problems across **5** 🌠 semantic-altering and **2** ⭐ semantic-preserving benchmarks
- 🔮 Allows evaluation/comparison across different **dimensions** and problem **types** (i.e., _Difficult_, _Creative_ or _Tool Use_ problems). See our [**visualization tool**](https://evo-eval.github.io/visualization.html) for ready-to-use comparison
- 🏆 Complete with [**leaderboard**](https://evo-eval.github.io/leaderboard.html), **groundtruth solutions**, **robust testcases** and **evaluation scripts** to easily fit into your evaluation pipeline
- 🤖 Generated LLM code samples from **>50** different models to save you time in running experiments

<sup>1</sup> coincidentally similar pronunciation with 😈 EvilEval

<p align="center">
<img src="./resources/example.gif" style="width:75%; margin-left: auto; margin-right: auto;">
</p>

Checkout our 📃 [paper](https://arxiv.org/abs/2403.19114) and [webpage](https://evo-eval.github.io) for more detail! 



## ⚡ Quick Start

Directly install the package:

```bash
pip install evoeval --upgrade
```

<details><summary>⏬ Nightly Version </summary>
<div>

```bash
pip install "git+https://github.com/evo-eval/evoeval.git" --upgrade
```

</div>
</details>

<details><summary>⏬ Local Repository</summary>
<div>

```bash
git clone https://github.com/evo-eval/evoeval.git
cd evoeval
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -r requirements.txt
```

</div>
</details>

Now you are ready to download EvoEval benchmarks and perform evaluation!

### 🧑‍💻 Code generation

To download our benchmarks, simply use the following code snippet:

```python
from evoeval.data import get_evo_eval

evoeval_benchmark = "EvoEval_difficult" # you can pick from 7 different benchmarks!

problems = get_evo_eval(evoeval_benchmark)
```

For code generation and evaluation, we adopt the same style as [HumanEval+](https://github.com/evalplus/evalplus) and [HumanEval](https://github.com/openai/human-eval).

Implement the `GEN_SOLUTION` function by calling the LLM to produce the complete solution (include the function header + code) and save the samples to `{benchmark}_samples.jsonl`:

```python
from evoeval.data import get_evo_eval, write_jsonl

evoeval_benchmark = "EvoEval_difficult"

samples = [
    dict(task_id=task_id, solution=GEN_SOLUTION(problem["prompt"]))
    for task_id, problem in get_evo_eval(evoeval_benchmark).items()
]
write_jsonl(f"{evoeval_benchmark}_samples.jsonl", samples)
```

> [!TIP]
> 
> EvoEval `samples.jsonl` expects the solution field to contain the **complete** code implementation, this is 
> slightly different from the original HumanEval where the solution field only contains the function body.
> 
> If you want to follow exactly like HumanEval setup, checkout our 🤗 Huggingface [datasets](https://huggingface.co/evoeval), which can be directly ran with
> HumanEval evaluation [script](https://huggingface.co/evoeval)

### 🕵️ Evaluation

You can use our provided [docker](https://docs.docker.com/get-docker/) image:

```bash
docker run --rm -v $(pwd):/app evoeval/evoeval:latest --dataset EvoEval_difficult --samples EvoEval_difficult_samples.jsonl
```

Or run it locally:

```bash
evoeval.evaluate --dataset EvoEval_difficult --samples EvoEval_difficult_samples.jsonl
```

Or if you are using it as a local repository:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python evoeval/evaluate.py --dataset EvoEval_difficult --samples EvoEval_difficult_samples.jsonl
```

You should expect to see the following output (when evaluated on GPT-4):
```
Computing expected output...
Expected outputs computed in 11.24s
Reading samples...
100it [00:00, 164.16it/s]
100%|████████████████████████████████████████████████████████████████| 100/100 [00:07<00:00, 12.77it/s]
EvoEval_difficult
pass@1: 0.520 # for reference GPT-4 solves more than 80% of problems in HumanEval
```
This shows the pass@1 score for the EvoEval_difficult benchmark. You can use `--i-just-wanna-run` to recompute the evaluation result

> [!Note]
> 
> You can also evaluate the LLM solutions in a folder format with each subfolder contains
> the LLM solution for each problem in the benchmark
>
> For example, you can grab the [GPT-4 solutions](https://github.com/evo-eval/evoeval/releases/download/v0.1.0/gpt-4_temp_0.0.zip) in our [v0.1.0 release](https://github.com/evo-eval/evoeval/releases/tag/v0.1.0).
> After unzipping, you can run the following command:
> 
> ```bash
> evoeval.evaluate --dataset EvoEval_difficult --samples gpt-4_temp_0.0/EvoEval_difficult 
> ```
>
> to obtain the same result as above using `.jsonl`


## 🔠 Benchmarks

**EvoEval** contains **7** different benchmarks, each with a unique set of problems 
evolved from the original **HumanEval** problems. 🌠 denotes semantic-altering benchmarks, 
while ⭐ denotes semantic-preserving benchmarks.:

<details><summary><b>🌠EvoEval_difficult:</b></summary>
<div>

> Introduce complexity by adding additional constraints and requirements,
> replace commonly used requirements to less common ones, or add additional reasoning
> steps to the original problem.
</div>
</details>

<details><summary><b>🌠EvoEval_creative:</b></summary>
<div>

> Generate a more creative problem compared to the original through the use
> of stories or uncommon narratives.
</div>
</details>


<details><summary><b>🌠EvoEval_subtle:</b></summary>
<div>

> Make a subtle and minor change to the original problem such as inverting or
> replacing a requirement.
</div>
</details>


<details><summary><b>🌠EvoEval_combine:</b></summary>
<div>

> Combine two different problems by integrating the concepts from both problems. In order to select problems that make sense to combine, we apply a simple heuristic
> to combine only problems of the same type together categorized based on the type of
> input arguments in the original problem.
</div>
</details>

<details><summary><b>🌠EvoEval_tool_use:</b></summary>
<div>

> Produce a new problem containing a main problem and one or more helpers
> functions which can be used to solve it. Each helper function is fully implemented and
> provides hints or useful functionality for solving the main problem. The main problem
> does not explicitly reference individual helper functions, and we do not require the model
> to use the provided helpers.
</div>
</details>


<details><summary><b>⭐EvoEval_verbose:</b></summary>
<div>

> Reword the original docstring to be more verbose. These verbose docstrings
> can use more descriptive language to illustrate the problem, include detailed explanation
> of the example output, and provide additional hints.
</div>
</details>

<details><summary><b>⭐EvoEval_concise:</b></summary>
<div>

> Reword the original docstring to be more concise by removing unnecessary
> details and using concise language. Furthermore, simple examples that are not required
> to demonstrate edge cases may be removed.

</div>
</details>

For each problem in each **EvoEval** benchmark, we include the complete groundtruth as well as test cases for functional evaluation.

> [!Note]
> 
> **Problem Structure**
> 
> ```json
> {
> "task_id": "identifier string for the task",
> "entry_point": "name of the function",
> "prompt": "function signature with docstring",
> "canonical_solution": "groundtruth implementation",
> "inputs": "test inputs for each problem",
> "parent": "original HumanEval problem it evolved from",
> "main": "special field of EvoEval_tool_use to show just the main problem description",
> "helpers": "special field of EvoEval_tool_use to show the helper functions"
> }
> ```

## 🤖 LLM Generated Code

To view the performance of **>50** LLMs on the EvoEval benchmarks,
we provide a complete [leaderboard](https://evo-eval.github.io/leaderboard.html) as well as a 
[visualization tool](https://evo-eval.github.io/visualization.html) to compare the performance of different models.

Further, we also provide all code samples from LLMs on the **EvoEval** benchmarks:

* See the attachment of our [v0.1.0 release](https://github.com/evo-eval/evoeval/releases/tag/v0.1.0).

Each LLM generation is packaged in a zip file named like `{model_name}_temp_0.0.zip`. You can unzip the folder and obtain the
LLM generation for each of our 7 benchmarks + the original HumanEval problems. Note that we only evaluate the greedy output for each LLM.

## 📝 Citation

```bibtex
@article{evoeval,
  author    = {Xia, Chunqiu Steven and Deng, Yinlin and Zhang, Lingming},
  title     = {Top Leaderboard Ranking = Top Coding Proficiency, Always? EvoEval: Evolving Coding Benchmarks via LLM},
  year      = {2024},
  journal   = {arXiv preprint},
}
```

> [!Note]
> 
> The first two authors contributed equally to this work, with author order determined via [_Nigiri_](https://senseis.xmp.net/?Nigiri)

## 🙏 Acknowledgement

* [HumanEval](https://github.com/openai/human-eval)
* We especially thank [EvalPlus](https://github.com/evalplus/evalplus)

## Harbor parity (this branch)

This branch (`harbor-parity-autoadapter`, based on `evo-eval/evoeval@d5ca310`) runs the original EvoEval pipeline for the parity experiment of the Harbor adapter `evoeval-autoadapter`. See all changes with `git diff d5ca310..harbor-parity-autoadapter`.

**Setup.** Generation uses Python 3.12 (same `openai` SDK as the Harbor side); evaluation uses Python 3.9, as EvoEval's evaluation image.

```bash
git clone -b harbor-parity-autoadapter https://github.com/MrLYG/evoeval.git && cd evoeval
uv venv .venv-gen --python 3.12
VIRTUAL_ENV=$PWD/.venv-gen uv pip install openai==3.46.0 tiktoken==0.14.0 rich==15.0.0 evalplus==0.2.0 .
uv venv .venv-eval --python 3.9
VIRTUAL_ENV=$PWD/.venv-eval uv pip install evalplus==0.2.0 numpy==2.0.2 .
export OPENAI_API_KEY=... OPENAI_BASE_URL=...   # OpenAI-compatible endpoint serving deepseek-v4.1-flash
```

**Run** (one run; use a new directory for each run):

```bash
GEN_PY=$PWD/.venv-gen/bin/python EVAL_PY=$PWD/.venv-eval/bin/python \
  harbor_parity/run_original_parity.sh runs/run1 25
cat runs/run1/summary.json     # pass@1 and per-task status
```

`run_original_parity.sh` calls, unchanged: `codegen/generate.py --model deepseek-v4.1-flash --bs 1 --temperature 0 --n_samples 1 --greedy` (in parallel over `--id-range` chunks), `tool/sanitize.py --eof`, and `python -m evoeval.evaluate`. `TASK_IDS="5 33 ..."` runs a subset through EvoEval's `EVOEVAL_OVERRIDE_PATH`.

**Settings used for parity** (same values on the Harbor side, agent `evoeval-codegen@1.0.0`):

| Setting | Value |
|---|---|
| Model | `deepseek-v4.1-flash` (served as `deepseek-v4-1-flash-261017`) |
| Prompt | original `OpenAIChatDecoder` prompt, system "You are a helpful assistant." |
| Decoding | `temperature=0`, `n=1`, greedy |
| `max_tokens` | prompt tokens + 32768 (`EVOEVAL_MAX_NEW_TOKENS`; original 512) |
| Per-request time limit | 900 s (`EVOEVAL_REQUEST_TIMEOUT`; original 100 s) |
| Prompt line endings | LF (`EVOEVAL_PROMPT_LF=1`) |
| Post-processing | `tool/sanitize.py --eof` |
| Grading | `evoeval.evaluate` defaults (`--min-time-limit 1`, `--gt-time-limit-factor 4.0`), evalplus 0.2.0 |

**Changes in this branch** (none touches the prompt text, the sanitizer or the grading):

- `codegen/model.py`: optional imports of torch / transformers / vllm / anthropic / google-generativeai, so the API path runs without them; a `deepseek-v4.1-flash` entry that uses `OpenAIChatDecoder` with `EVOEVAL_MAX_NEW_TOKENS` (default 32768). The model is a reasoning model; with 512 new tokens most replies end before any answer.
- `evoeval/util/api_request.py`: optional Google imports; the per-request alarm reads `EVOEVAL_REQUEST_TIMEOUT` (default 100, as before). Long reasoning requests exceed 100 s and would be retried forever.
- `codegen/generate.py`: with `EVOEVAL_PROMPT_LF=1`, CRLF line endings in the prompt (10 of the 100 `EvoEval_difficult` prompts) are sent as LF, as Harbor agents receive them. Off by default.
- `harbor_parity/run_original_parity.sh`, `harbor_parity/summarize.py`: the runner and the per-run summary.

Without these environment variables, the existing models and the pipeline behave exactly as upstream.

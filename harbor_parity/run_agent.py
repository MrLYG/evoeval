"""Run a CLI coding agent (mini-swe-agent) on EvoEval problems, original-benchmark side.

EvoEval's own harness prompts a model once and scores the extracted function.
For parity with the Harbor adapter (harbor-framework/adapters, ``evoeval-auto``)
the same agent must run on both sides. This script does, for every problem:

1. render the task instruction from ``instruction_template.md`` (the EvoEval
   API-model prompt "Please complete the following code snippet." + where to
   save the answer) - byte-identical to the Harbor task's ``instruction.md``;
2. start a Daytona sandbox built from ``Dockerfile`` (identical to the Harbor
   task environment: python:3.10.22-slim + numpy 2.2.6), 2 CPUs / 4 GB / 10 GB;
3. install and run mini-swe-agent exactly as Harbor's ``mini-swe-agent`` agent
   does (same install command, same CLI flags, same environment variables,
   working directory /workspace, same agent timeout). Like the Harbor task
   (``[agent] network_mode = "no-network"``), the agent has no network while it
   runs, except the hosts given with ``--allow-agent-host`` (its model API);
   the install step keeps network access;
4. download ``/workspace/solution.py`` and the agent logs, delete the sandbox.

It writes ``<run-dir>/samples.jsonl`` with one ``{"task_id", "solution"}`` line
per EvoEval problem (the file content as the ``solution``; an empty string if
the agent wrote no file, or for problems outside ``--task-ids``). Score it with
this fork's evaluator (upstream plus the documented float fix)::

    OPENBLAS_NUM_THREADS=1 evoeval.evaluate --dataset EvoEval_difficult \
        --samples <run-dir>/samples.jsonl --parallel 8

Infrastructure failures (sandbox, install or network-policy errors) are recorded in
``<run-dir>/tasks/<id>/meta.json`` with ``"infra_error"`` and are NOT written
to samples.jsonl until they are re-run (``--resume`` re-runs only problems
without a completed attempt).
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    DaytonaConfig,
    Image,
    Resources,
    SessionExecuteRequest,
)

from evoeval.data import get_evo_eval

HERE = Path(__file__).resolve().parent
DOCKERFILE = HERE / "Dockerfile"
TEMPLATE = HERE / "instruction_template.md"

# ---- Harbor 0.23.0 mini-swe-agent agent, reproduced -------------------------
MSWEA_PYTHON = "3.12"  # harbor: _MINI_SWE_AGENT_PYTHON_VERSION
LITELLM_TIMEOUT_SEC = 3600  # harbor: _DEFAULT_LITELLM_TIMEOUT_SEC
TRAJ_PATH = "/logs/agent/mini-swe-agent.trajectory.json"
LOG_PATH = "/logs/agent/mini-swe-agent.txt"
SOURCE_PATH_ENV = (
    'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
    'else export PATH="$HOME/.local/bin:$PATH"; fi; '
)
# harbor: BaseInstalledAgent.ensure_system_dependencies(("curl","bash","build_tools","git"))
SYSDEP_CHECK = (
    "command -v curl >/dev/null 2>&1 && command -v bash >/dev/null 2>&1 && "
    "command -v gcc >/dev/null 2>&1 && command -v make >/dev/null 2>&1 && "
    "command -v git >/dev/null 2>&1"
)
SYSDEP_INSTALL = "apt-get update && apt-get install -y curl bash build-essential git"


def install_command(version: str) -> str:
    """harbor MiniSweAgent.install()."""
    version_spec = f"=={version}" if version else ""
    py = shlex.quote(MSWEA_PYTHON)
    return (
        "set -euo pipefail; "
        "if ! command -v uv >/dev/null 2>&1; then"
        "  curl -LsSf https://astral.sh/uv/install.sh | sh;"
        " fi && "
        'if ! grep -q \'export PATH="$HOME/.local/bin:$PATH"\' "$HOME/.bashrc" 2>/dev/null; then'
        '  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> "$HOME/.bashrc";'
        " fi && "
        'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
        'export PATH="$HOME/.local/bin:$PATH" && '
        f"uv python install {py} && "
        f"uv tool install --python {py} "
        f"mini-swe-agent{version_spec} "
        "--with litellm --with orjson --with fastapi && "
        "mini-swe-agent --help"
    )


def run_command(instruction: str, model: str, session_id: str, max_tokens: int | None) -> str:
    """harbor MiniSweAgent.run() (no custom config, no reasoning_effort)."""
    config_flags = f"-c model.model_kwargs.timeout={LITELLM_TIMEOUT_SEC} "
    config_flags += (
        "-c "
        + shlex.quote(f"model.model_kwargs.extra_headers.X-Session-ID={session_id}")
        + " "
    )
    if max_tokens is not None:
        config_flags += f"-c model.model_kwargs.max_tokens={max_tokens} "
    config_flags = "-c mini " + config_flags
    extra_flags = "--cost-limit 0 "  # harbor MiniSweAgentOptions.cost_limit default "0"
    return (
        SOURCE_PATH_ENV
        + f"mini-swe-agent --yolo --model={model} --task={shlex.quote(instruction)} "
        f"--output={TRAJ_PATH} {extra_flags}"
        f"{config_flags}"
        f"--exit-immediately 2>&1 </dev/null | tee {LOG_PATH}"
    )


def render_instruction(problem: dict) -> str:
    """Same rendering as the Harbor adapter (evoeval_auto.adapter)."""
    text = TEMPLATE.read_text()
    for key, value in (
        ("ENTRY_POINT", problem["entry_point"]),
        ("PROMPT", problem["prompt"].strip()),
    ):
        text = text.replace("{{" + key + "}}", value)
    return text


def load_env_file(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ---- Daytona helpers ----------------------------------------------------------
async def sh(sandbox, command: str, *, env: dict[str, str] | None = None,
             cwd: str | None = "/workspace", timeout: int | None = None):
    """Run ``command`` like harbor's Daytona exec: ``cd <cwd> && env K=V bash -c '<cmd>'``.

    Returns (exit_code or None on timeout, stdout, stderr).
    """
    session_id = str(uuid.uuid4())
    await sandbox.process.create_session(session_id)
    full = f"bash -c {shlex.quote('set -o pipefail; ' + command)}"
    if env:
        full = "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + full
    if cwd:
        full = f"cd {cwd} && {full}"
    resp = await sandbox.process.execute_session_command(
        session_id, SessionExecuteRequest(command=full, run_async=True)
    )
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        cmd = await sandbox.process.get_session_command(session_id, resp.cmd_id)
        if cmd.exit_code is not None:
            break
        if deadline is not None and time.monotonic() > deadline:
            # Deleting the session kills its processes, so nothing keeps writing
            # after the cutoff (e.g. the agent writing /workspace/solution.py).
            # If that cannot be done or confirmed, the attempt is an
            # infrastructure error (re-run), never a scored submission.
            await sandbox.process.delete_session(session_id)
            await asyncio.sleep(2)
            await ensure_agent_stopped(sandbox)
            return None, "", f"timed out after {timeout}s"
        await asyncio.sleep(1)
    logs = await sandbox.process.get_session_command_logs(session_id, resp.cmd_id)
    return int(cmd.exit_code), logs.stdout or "", logs.stderr or ""


AGENT_ALIVE_CHECK = (
    "for p in /proc/[0-9]*; do "
    "if tr '\\0' ' ' < $p/cmdline 2>/dev/null | grep -q 'mini-swe-agen[t]'; then echo alive; fi; "
    "done"
)


async def ensure_agent_stopped(sandbox) -> None:
    """Raise unless no mini-swe-agent process is left in the sandbox."""
    for attempt in range(5):
        code, out, _ = await sh(sandbox, AGENT_ALIVE_CHECK, cwd=None, timeout=60)
        if code == 0 and "alive" not in out:
            return
        if code == 0:
            await sh(sandbox,
                     "for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null | "
                     "grep -q 'mini-swe-agen[t]' && kill -9 ${p#/proc/} 2>/dev/null; done; true",
                     cwd=None, timeout=60)
        await asyncio.sleep(2)
    raise RuntimeError("timed-out agent could not be confirmed stopped")


def agent_network_kwargs(hosts: list[str]) -> dict:
    """Daytona network settings for the agent phase, as Harbor 0.23.0 applies them.

    The task sets ``[agent] network_mode = "no-network"``; ``--allow-agent-host``
    turns that into an allowlist of those hosts only
    (``merge_extra_allowlists`` and ``DaytonaEnvironment._network_kwargs`` with
    ``clear_public_allowlist=True``).
    """
    if not hosts:
        return {"network_block_all": True}
    domains, cidrs = [], []
    for h in hosts:
        try:
            cidrs.append(str(ipaddress.ip_network(h, strict=False)))
        except ValueError:
            domains.append(h)
    if domains:
        return {"network_block_all": False, "domain_allow_list": ",".join(domains), "network_allow_list": ""}
    return {"network_block_all": False, "network_allow_list": ",".join(cidrs), "domain_allow_list": ""}


# Harbor restores the environment baseline (public) after the agent phase.
PUBLIC_NETWORK = {"network_block_all": False, "domain_allow_list": "", "network_allow_list": ""}


async def set_network(sandbox, kwargs: dict) -> None:
    """harbor DaytonaEnvironment._apply_network_policy: 3 attempts, exponential wait."""
    for attempt in range(3):
        try:
            await sandbox.update_network_settings(**kwargs)
            return
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(min(10, 2 ** attempt))


async def file_exists(sandbox, path: str) -> bool:
    """True/False from ``test -f``; raises if the check itself cannot run."""
    for attempt in range(3):
        code, _, _ = await sh(sandbox, f"test -f {shlex.quote(path)}", cwd=None, timeout=60)
        if code in (0, 1):
            return code == 0
        await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not check {path}")


async def download(sandbox, path: str) -> bytes:
    """Download a file that exists; transport errors are retried, then raised."""
    last = None
    for attempt in range(3):
        try:
            return await sandbox.fs.download_file(path)
        except Exception as exc:  # transient transport error
            last = exc
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download of {path} failed: {last}")


async def download_log(sandbox, path: str) -> bytes | None:
    """Best-effort download of an agent log (not part of the score)."""
    try:
        return await download(sandbox, path) if await file_exists(sandbox, path) else None
    except Exception:
        return None


async def run_one(daytona, problem: dict, args, env_vars: dict[str, str], out_dir: Path, sem):
    task_id = problem["task_id"]
    tdir = out_dir / "tasks" / task_id.replace("/", "_")
    tdir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"task_id": task_id, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    instruction = render_instruction(problem)
    (tdir / "instruction.md").write_text(instruction)
    # harbor: X-Session-ID = "<trial name>__agent"; here "<task>__<run>__agent"
    session_id = f"{task_id.replace('/', '-').lower()}__{args.run_name}__agent"
    async with sem:
        sandbox = None
        try:
            params = CreateSandboxFromImageParams(
                image=Image.from_dockerfile(str(DOCKERFILE)),
                resources=Resources(cpu=args.cpus, memory=args.memory_gb, disk=args.disk_gb),
                auto_stop_interval=0,
                auto_delete_interval=0,
                labels={"evoeval-parity": "original", "evoeval-parity-run": args.run_name},
            )
            sandbox = await daytona.create(params=params, timeout=args.build_timeout)
            meta["sandbox_id"] = sandbox.id
            # harbor: ensure workdir + log mount targets
            await sh(sandbox, "mkdir -p /workspace /logs/agent /logs/verifier", cwd=None)

            # ---- agent setup (harbor: setup timeout 360 s by default) ----
            # harbor applies --ae/extra_env to every agent setup and run command.
            agent_env = dict(args.agent_env)
            t0 = time.time()
            code, _, _ = await sh(sandbox, SYSDEP_CHECK, env=agent_env or None, cwd=None)
            if code != 0:
                await sh(sandbox, SYSDEP_INSTALL, env={**agent_env, "DEBIAN_FRONTEND": "noninteractive"}, cwd=None)
            code, out, err = await sh(sandbox, install_command(args.agent_version), env=agent_env or None,
                                      timeout=args.setup_timeout)
            meta["setup_seconds"] = round(time.time() - t0, 1)
            if code != 0:
                (tdir / "setup.log").write_text(out + "\n" + err)
                raise RuntimeError(f"agent install failed (exit {code})")
            code, out, _ = await sh(
                sandbox, SOURCE_PATH_ENV + "uv tool list 2>/dev/null | grep mini-swe-agent"
            )
            meta["agent_version"] = out.strip()
            code, out, _ = await sh(
                sandbox,
                SOURCE_PATH_ENV + '"$(uv tool dir)/mini-swe-agent/bin/python" -c '
                + shlex.quote("import importlib.metadata as m; print(m.version('litellm'))"),
            )
            meta["litellm_version"] = out.strip()

            # ---- agent run, with the agent-phase network policy ----
            # A failure to apply it is an infrastructure error: the agent never
            # runs with open network.
            await set_network(sandbox, agent_network_kwargs(args.allow_agent_host))
            meta["agent_network"] = "allowlist" if args.allow_agent_host else "no-network"
            run_env = {
                **agent_env,
                **env_vars,
                "MSWEA_CONFIGURED": "true",
                "MSWEA_COST_TRACKING": "ignore_errors",
            }
            t1 = time.time()
            code, _, err = await sh(
                sandbox,
                run_command(instruction, args.model, session_id, args.max_tokens),
                env=run_env,
                timeout=args.agent_timeout,
            )
            meta["agent_seconds"] = round(time.time() - t1, 1)
            meta["agent_exit_code"] = code
            meta["agent_timed_out"] = code is None
            await set_network(sandbox, PUBLIC_NETWORK)

            # ---- collect ----
            # A missing file is the agent's result (empty solution); a failed
            # check or download is an infrastructure error (re-run with --resume).
            sol = None
            if await file_exists(sandbox, "/workspace/solution.py"):
                sol = await download(sandbox, "/workspace/solution.py")
            meta["solution_found"] = sol is not None
            solution = sol.decode("utf-8", errors="replace") if sol is not None else ""
            (tdir / "solution.py").write_text(solution)
            for remote, local in ((TRAJ_PATH, "mini-swe-agent.trajectory.json"), (LOG_PATH, "mini-swe-agent.txt")):
                data = await download_log(sandbox, remote)
                if data is not None:
                    (tdir / local).write_bytes(data)
            meta["completed"] = True
            meta["infra_error"] = None
        except Exception as exc:  # infrastructure failure: no sample, re-run later
            meta["completed"] = False
            meta["infra_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if sandbox is not None:
                try:
                    await daytona.delete(sandbox)
                except Exception as exc:
                    meta["delete_error"] = str(exc)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (tdir / "meta.json").write_text(json.dumps(meta, indent=2))
    status = "ok" if meta.get("completed") else f"INFRA ERROR {meta['infra_error']}"
    print(f"[{time.strftime('%H:%M:%S')}] {task_id}: {status} "
          f"(solution={meta.get('solution_found')}, agent_s={meta.get('agent_seconds')})", flush=True)
    return meta


def write_samples(problems: dict, out_dir: Path) -> tuple[int, list[str]]:
    """samples.jsonl over ALL problems (upstream evaluate requires every task).

    evoeval.evaluate reuses an existing ``samples_eval_results.json`` without
    checking the samples, so it is deleted whenever samples.jsonl is rewritten.
    """
    stale = out_dir / "samples_eval_results.json"
    if stale.exists():
        stale.unlink()
    done, pending = 0, []
    selected = selected_ids(out_dir)
    with open(out_dir / "samples.jsonl", "w") as f:
        for task_id in problems:
            tdir = out_dir / "tasks" / task_id.replace("/", "_")
            meta_p = tdir / "meta.json"
            solution = ""
            if meta_p.exists() and json.loads(meta_p.read_text()).get("completed"):
                solution = (tdir / "solution.py").read_text()
                done += 1
            elif task_id in selected:
                pending.append(task_id)
            f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
    return done, pending


IMMUTABLE_SETTINGS = (
    "dataset", "model", "agent_version", "max_tokens", "agent_timeout", "setup_timeout",
    "build_timeout", "cpus", "memory_gb", "disk_gb", "allow_agent_host",
)


def selected_ids(out_dir: Path) -> set[str]:
    p = out_dir / "selected_task_ids.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


async def main_async(args) -> int:
    problems = get_evo_eval(args.dataset)
    ids = list(problems)
    if args.task_ids:
        wanted = set(args.task_ids)
        ids = [t for t in ids if t in wanted]
        if len(ids) != len(wanted):
            print(f"unknown task ids: {sorted(wanted - set(ids))}", file=sys.stderr)
            return 2
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_p = out_dir / "selected_task_ids.json"
    cfg_p = out_dir / "run_config.json"
    config = {k: getattr(args, k) for k in IMMUTABLE_SETTINGS}
    config["agent_env"] = [list(kv) for kv in args.agent_env]
    if sel_p.exists() or cfg_p.exists():
        # An existing run directory may only be resumed with identical settings
        # and the same task selection; otherwise use a new --run-dir.
        old_ids = json.loads(sel_p.read_text()) if sel_p.exists() else None
        old_cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else None
        if not args.resume or old_ids != ids or old_cfg != config:
            print(f"{out_dir} already holds a run with different settings or tasks "
                  "(or --resume was not given); use a new --run-dir", file=sys.stderr)
            return 2
    else:
        sel_p.write_text(json.dumps(ids))
        cfg_p.write_text(json.dumps(config, indent=2))

    todo = []
    for t in ids:
        meta_p = out_dir / "tasks" / t.replace("/", "_") / "meta.json"
        if args.resume and meta_p.exists() and json.loads(meta_p.read_text()).get("completed"):
            continue
        todo.append(problems[t])
    print(f"{len(todo)} problem(s) to run ({len(ids)} selected)", flush=True)

    file_env = load_env_file(Path(args.env_file)) if args.env_file else {}
    key = file_env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = file_env.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    dkey = file_env.get("DAYTONA_API_KEY") or os.environ.get("DAYTONA_API_KEY")
    env_vars = {"OPENAI_API_KEY": key}
    if base:
        env_vars["OPENAI_BASE_URL"] = base
        env_vars["OPENAI_API_BASE"] = base

    daytona = AsyncDaytona(DaytonaConfig(api_key=dkey))
    sem = asyncio.Semaphore(args.n_concurrent)
    try:
        await asyncio.gather(*(run_one(daytona, p, args, env_vars, out_dir, sem) for p in todo))
    finally:
        await daytona.close()

    done, pending = write_samples(problems, out_dir)
    print(f"samples.jsonl written: {done} completed, {len(pending)} pending (infra errors): {pending}")
    return 0 if not pending else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="EvoEval_difficult")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-name", default=None, help="label for sandboxes/session ids (default: run-dir name)")
    p.add_argument("--task-ids", nargs="+", default=None, help="e.g. EvoEval/0 EvoEval/7 (default: all)")
    p.add_argument("--model", default="openai/deepseek-v4.1-flash")
    p.add_argument("--agent-version", default="2.4.6")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--agent-timeout", type=int, default=1800, help="task.toml [agent].timeout_sec")
    p.add_argument("--setup-timeout", type=int, default=360)
    p.add_argument("--build-timeout", type=int, default=1200, help="task.toml [environment].build_timeout_sec")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--memory-gb", type=int, default=4)
    p.add_argument("--disk-gb", type=int, default=10)
    p.add_argument("--n-concurrent", type=int, default=6)
    p.add_argument("--env-file", default=None, help="file with OPENAI_API_KEY, OPENAI_BASE_URL, DAYTONA_API_KEY")
    p.add_argument("--resume", action="store_true", help="skip problems that already completed")
    p.add_argument("--agent-env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra env for agent setup and run (harbor --ae), e.g. "
                        "UV_EXCLUDE_NEWER=2026-10-17T00:00:00Z")
    p.add_argument("--allow-agent-host", action="append", default=[], metavar="HOST",
                   help="host (or IPv4/CIDR) the agent may reach while it runs, e.g. its model API "
                        "(harbor --allow-agent-host). The agent has no other network access.")
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = Path(args.run_dir).name
    args.agent_env = [tuple(kv.split("=", 1)) for kv in args.agent_env]
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

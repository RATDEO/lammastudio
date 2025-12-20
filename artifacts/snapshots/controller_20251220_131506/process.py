"""Process management for vLLM/SGLang."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

from .backends import build_sglang_command, build_vllm_command
from .config import settings
from .models import Backend, ProcessInfo, Recipe


def _extract_flag(cmdline: List[str], flag: str) -> Optional[str]:
    """Extract value of a CLI flag."""
    for i, arg in enumerate(cmdline):
        if arg == flag and i + 1 < len(cmdline):
            return cmdline[i + 1]
    return None


def _build_env(recipe: Recipe) -> Dict[str, str]:
    env = os.environ.copy()

    env_vars = (
        recipe.extra_args.get("env_vars")
        or recipe.extra_args.get("env-vars")
        or recipe.extra_args.get("envVars")
    )
    if isinstance(env_vars, dict):
        for k, v in env_vars.items():
            if v is None:
                continue
            env[str(k)] = str(v)

    cuda_visible_devices = (
        recipe.extra_args.get("cuda_visible_devices")
        or recipe.extra_args.get("cuda-visible-devices")
        or recipe.extra_args.get("CUDA_VISIBLE_DEVICES")
    )
    if cuda_visible_devices not in (None, "", False):
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    return env


def _is_inference_process(cmdline: List[str]) -> Optional[str]:
    """Check if cmdline is vLLM or SGLang, return backend name."""
    if not cmdline:
        return None
    joined = " ".join(cmdline)
    if "vllm.entrypoints.openai.api_server" in joined:
        return "vllm"
    if len(cmdline) >= 2 and cmdline[0].endswith("vllm") and cmdline[1] == "serve":
        return "vllm"
    if "sglang.launch_server" in joined:
        return "sglang"
    return None


def find_inference_process(port: int) -> Optional[ProcessInfo]:
    """Find running inference process on given port."""
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            backend = _is_inference_process(cmdline)
            if not backend:
                continue
            p = _extract_flag(cmdline, "--port")
            if p is None or int(p) != port:
                continue
            # Extract model path
            model_path = _extract_flag(cmdline, "--model") or _extract_flag(cmdline, "--model-path")
            if not model_path and len(cmdline) >= 3 and cmdline[1] == "serve":
                model_path = cmdline[2] if not cmdline[2].startswith("-") else None
            return ProcessInfo(
                pid=proc.info["pid"],
                backend=backend,
                model_path=model_path,
                port=port,
                served_model_name=_extract_flag(cmdline, "--served-model-name"),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue
    return None


async def kill_process(pid: int, force: bool = False) -> bool:
    """Kill process and its children."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True

    # Kill children first
    for child in proc.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    # Terminate main process
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except psutil.TimeoutExpired:
        if force:
            proc.kill()
    except psutil.NoSuchProcess:
        pass

    await asyncio.sleep(1)
    return True


async def launch_model(recipe: Recipe) -> Tuple[bool, Optional[int], str]:
    """Launch inference server with recipe config."""
    recipe.port = settings.inference_port  # Override with configured port

    if recipe.backend == Backend.SGLANG:
        cmd = build_sglang_command(recipe)
    else:
        cmd = build_vllm_command(recipe)

    log_file = Path(f"/tmp/vllm_{recipe.id}.log")
    env = _build_env(recipe)

    try:
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        await asyncio.sleep(3)

        if proc.poll() is not None:
            tail = log_file.read_text()[-500:] if log_file.exists() else ""
            return False, None, f"Process exited early: {tail}"

        return True, proc.pid, str(log_file)
    except Exception as e:
        return False, None, str(e)


async def evict_model(force: bool = False) -> Optional[int]:
    """Stop current running model."""
    current = find_inference_process(settings.inference_port)
    if not current:
        return None
    await kill_process(current.pid, force=force)
    return current.pid


async def switch_model(recipe: Recipe, force: bool = False) -> Tuple[bool, Optional[int], str]:
    """Switch to a new model (evict current + launch new)."""
    await evict_model(force=force)
    await asyncio.sleep(2)
    return await launch_model(recipe)

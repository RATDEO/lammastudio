"""FastAPI application - minimal controller API."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections import deque
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .gpu import get_gpu_info
from .models import HealthResponse, LaunchResult, OpenAIModelInfo, OpenAIModelList, Recipe
from .process import evict_model, find_inference_process, switch_model
from .store import RecipeStore

app = FastAPI(
    title="vLLM Studio Controller",
    version=__version__,
    description="Minimal model lifecycle management for vLLM/SGLang",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_store: Optional[RecipeStore] = None
_switch_lock = asyncio.Lock()


def get_store() -> RecipeStore:
    global _store
    if _store is None:
        _store = RecipeStore(settings.db_path)
    return _store


# --- Authentication middleware ---
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not settings.api_key:
        return await call_next(request)

    if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != settings.api_key:
        return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})

    return await call_next(request)


# --- Health ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check."""
    current = find_inference_process(settings.inference_port)
    inference_ready = False

    if current:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"http://localhost:{settings.inference_port}/health")
                inference_ready = r.status_code == 200
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        version=__version__,
        inference_ready=inference_ready,
        backend_reachable=inference_ready,
        running_model=current.served_model_name or current.model_path if current else None,
    )


@app.get("/status", tags=["System"])
async def status():
    """Detailed status."""
    current = find_inference_process(settings.inference_port)
    return {
        "running": current is not None,
        "process": current.model_dump() if current else None,
        "inference_port": settings.inference_port,
    }


@app.get("/gpus", tags=["System"])
async def gpus():
    """Get GPU information."""
    gpu_list = get_gpu_info()
    return {
        "count": len(gpu_list),
        "gpus": [gpu.model_dump() for gpu in gpu_list],
    }


# --- OpenAI-compatible endpoints ---
@app.get("/v1/models", response_model=OpenAIModelList, tags=["OpenAI Compatible"])
async def list_models_openai(store: RecipeStore = Depends(get_store)):
    """List all models in OpenAI format."""
    import time

    recipes = store.list()
    current = find_inference_process(settings.inference_port)

    # Get active model metadata from vLLM/SGLang if available
    active_model_data = None
    if current:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"http://localhost:{settings.inference_port}/v1/models")
                if r.status_code == 200:
                    active_model_data = r.json()
        except Exception:
            pass

    models = []
    current_time = int(time.time())

    for recipe in recipes:
        is_active = False
        max_model_len = recipe.max_model_len

        # Check if this recipe is the currently running model
        if current and current.model_path and recipe.model_path in current.model_path:
            is_active = True
            # Try to get max_model_len from the active model's endpoint
            if active_model_data and "data" in active_model_data:
                for model in active_model_data["data"]:
                    if "max_model_len" in model:
                        max_model_len = model["max_model_len"]
                        break

        # Use served_model_name if available, otherwise use recipe ID
        model_id = recipe.served_model_name or recipe.id

        models.append(
            OpenAIModelInfo(
                id=model_id,
                created=current_time,
                active=is_active,
                max_model_len=max_model_len,
            )
        )

    return OpenAIModelList(data=models)


@app.get("/v1/models/{model_id}", response_model=OpenAIModelInfo, tags=["OpenAI Compatible"])
async def get_model_openai(model_id: str, store: RecipeStore = Depends(get_store)):
    """Get a specific model in OpenAI format."""
    import time

    # Try to find by served_model_name first, then by ID
    recipes = store.list()
    recipe = None
    for r in recipes:
        if (r.served_model_name and r.served_model_name == model_id) or r.id == model_id:
            recipe = r
            break

    if not recipe:
        raise HTTPException(status_code=404, detail="Model not found")

    current = find_inference_process(settings.inference_port)
    is_active = False
    max_model_len = recipe.max_model_len

    # Check if this is the active model and get metadata
    if current and current.model_path and recipe.model_path in current.model_path:
        is_active = True
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"http://localhost:{settings.inference_port}/v1/models")
                if r.status_code == 200:
                    active_model_data = r.json()
                    if "data" in active_model_data:
                        for model in active_model_data["data"]:
                            if "max_model_len" in model:
                                max_model_len = model["max_model_len"]
                                break
        except Exception:
            pass

    # Use served_model_name if available, otherwise use recipe ID
    display_id = recipe.served_model_name or recipe.id

    return OpenAIModelInfo(
        id=display_id,
        created=int(time.time()),
        active=is_active,
        max_model_len=max_model_len,
    )


# --- Recipes ---
@app.get("/recipes", tags=["Recipes"])
async def list_recipes(store: RecipeStore = Depends(get_store)):
    """List all recipes."""
    recipes = store.list()
    current = find_inference_process(settings.inference_port)
    result = []
    for r in recipes:
        status = "stopped"
        if current and current.model_path and r.model_path in current.model_path:
            status = "running"
        result.append({**r.model_dump(), "status": status})
    return result


@app.get("/recipes/{recipe_id}", tags=["Recipes"])
async def get_recipe(recipe_id: str, store: RecipeStore = Depends(get_store)):
    """Get a recipe by ID."""
    recipe = store.get(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe


@app.post("/recipes", tags=["Recipes"])
async def create_recipe(recipe: Recipe, store: RecipeStore = Depends(get_store)):
    """Create or update a recipe."""
    store.save(recipe)
    return {"success": True, "id": recipe.id}


@app.put("/recipes/{recipe_id}", tags=["Recipes"])
async def update_recipe(recipe_id: str, recipe: Recipe, store: RecipeStore = Depends(get_store)):
    """Update a recipe by ID."""
    if recipe.id != recipe_id:
        recipe.id = recipe_id
    store.save(recipe)
    return {"success": True, "id": recipe.id}


@app.delete("/recipes/{recipe_id}", tags=["Recipes"])
async def delete_recipe(recipe_id: str, store: RecipeStore = Depends(get_store)):
    """Delete a recipe."""
    if not store.delete(recipe_id):
        raise HTTPException(status_code=404, detail="Recipe not found")
    return {"success": True}


# --- Model lifecycle ---
@app.post("/launch/{recipe_id}", response_model=LaunchResult, tags=["Lifecycle"])
async def launch(recipe_id: str, force: bool = False, store: RecipeStore = Depends(get_store)):
    """Launch a model by recipe ID."""
    recipe = store.get(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    async with _switch_lock:
        success, pid, message = await switch_model(recipe, force=force)

    return LaunchResult(
        success=success,
        pid=pid,
        message=message,
        log_file=f"/tmp/vllm_{recipe_id}.log" if success else None,
    )


@app.post("/evict", tags=["Lifecycle"])
async def evict(force: bool = False):
    """Stop the running model."""
    async with _switch_lock:
        pid = await evict_model(force=force)
    return {"success": True, "evicted_pid": pid}


@app.get("/wait-ready", tags=["Lifecycle"])
async def wait_ready(timeout: int = 300):
    """Wait for inference backend to be ready."""
    import time

    start = time.time()
    while time.time() - start < timeout:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"http://localhost:{settings.inference_port}/health")
                if r.status_code == 200:
                    return {"ready": True, "elapsed": int(time.time() - start)}
        except Exception:
            pass
        await asyncio.sleep(2)

    return {"ready": False, "elapsed": timeout, "error": "Timeout waiting for backend"}


# --- Logs ---
def _log_path_for(session_id: str) -> Path:
    safe = "".join(ch for ch in (session_id or "") if ch.isalnum() or ch in ("-", "_", "."))
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid log session id")
    return Path("/tmp") / f"vllm_{safe}.log"


def _tail_lines(path: Path, limit: int) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=limit))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log not found")


@app.get("/logs", tags=["Logs"])
async def list_logs(store: RecipeStore = Depends(get_store)):
    """List available inference log files."""
    current = find_inference_process(settings.inference_port)
    log_files = sorted(Path("/tmp").glob("vllm_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    sessions = []
    for p in log_files:
        sid = p.name.removeprefix("vllm_").removesuffix(".log")
        recipe = store.get(sid)
        created_at = dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).isoformat()

        status = "stopped"
        if current and recipe and current.model_path and recipe.model_path and recipe.model_path in current.model_path:
            status = "running"
        elif current and recipe and current.served_model_name and recipe.served_model_name == current.served_model_name:
            status = "running"

        sessions.append(
            {
                "id": sid,
                "recipe_id": recipe.id if recipe else sid,
                "recipe_name": recipe.name if recipe else None,
                "model_path": recipe.model_path if recipe else None,
                "model": (recipe.served_model_name or recipe.name) if recipe else sid,
                "backend": recipe.backend.value if recipe else None,
                "created_at": created_at,
                "status": status,
            }
        )

    return {"sessions": sessions}


@app.get("/logs/{session_id}", tags=["Logs"])
async def get_logs(session_id: str, limit: int = 2000):
    """Get log content for a session (returns both `logs` and `content` for UI compatibility)."""
    limit = max(1, min(int(limit), 20000))
    path = _log_path_for(session_id)
    lines = _tail_lines(path, limit)
    logs = [ln.rstrip("\n") for ln in lines]
    return {"id": session_id, "logs": logs, "content": "\n".join(logs)}


@app.delete("/logs/{session_id}", tags=["Logs"])
async def delete_logs(session_id: str):
    """Delete a log file."""
    path = _log_path_for(session_id)
    try:
        path.unlink()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log not found")
    return {"success": True}


# --- MCP (minimal built-in tools) ---
_MCP_CFG_NAME = "mcp_servers.json"


def _mcp_cfg_path() -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / _MCP_CFG_NAME


def _read_mcp_servers() -> list[dict]:
    path = _mcp_cfg_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [s for s in data if isinstance(s, dict)]
    except Exception:
        return []
    return []


def _write_mcp_servers(servers: list[dict]) -> None:
    _mcp_cfg_path().write_text(json.dumps(servers, indent=2, sort_keys=True), encoding="utf-8")


@app.get("/mcp/servers", tags=["MCP"])
async def list_mcp_servers():
    return _read_mcp_servers()


@app.post("/mcp/servers", tags=["MCP"])
async def add_mcp_server(server: dict):
    name = str(server.get("name") or "").strip()
    command = str(server.get("command") or "").strip()
    if not name or not command:
        raise HTTPException(status_code=400, detail="`name` and `command` required")

    servers = [s for s in _read_mcp_servers() if s.get("name") != name]
    server["name"] = name
    server["command"] = command
    server["enabled"] = bool(server.get("enabled", True))
    server["args"] = list(server.get("args") or [])
    server["env"] = dict(server.get("env") or {})
    servers.append(server)
    _write_mcp_servers(servers)
    return {"success": True}


@app.put("/mcp/servers/{name}", tags=["MCP"])
async def update_mcp_server(name: str, server: dict):
    server["name"] = name
    return await add_mcp_server(server)


@app.delete("/mcp/servers/{name}", tags=["MCP"])
async def delete_mcp_server(name: str):
    servers = _read_mcp_servers()
    next_servers = [s for s in servers if s.get("name") != name]
    if len(next_servers) == len(servers):
        raise HTTPException(status_code=404, detail="Server not found")
    _write_mcp_servers(next_servers)
    return {"success": True}


@app.get("/mcp/tools", tags=["MCP"])
async def list_mcp_tools():
    # Provide a small built-in tool set so tool-calling works out of the box.
    return {
        "tools": [
            {
                "server": "builtin",
                "name": "time",
                "description": "Get the current time (UTC) as an ISO 8601 string.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "server": "builtin",
                "name": "fetch",
                "description": "Fetch a URL via HTTP GET and return text (truncated).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "max_bytes": {"type": "integer", "description": "Max bytes to return", "default": 20000},
                        "timeout_sec": {"type": "number", "description": "Request timeout seconds", "default": 20},
                        "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        ]
    }


@app.post("/mcp/tools/{server}/{tool_name}", tags=["MCP"])
async def call_mcp_tool(server: str, tool_name: str, payload: dict):
    if server != "builtin":
        raise HTTPException(status_code=404, detail="Unknown MCP server")

    if tool_name == "time":
        return {"result": {"utc": dt.datetime.now(tz=dt.timezone.utc).isoformat()}}

    if tool_name == "fetch":
        url = str(payload.get("url") or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="`url` required")
        max_bytes = int(payload.get("max_bytes") or 20000)
        max_bytes = max(1, min(max_bytes, 250_000))
        timeout_sec = float(payload.get("timeout_sec") or 20)
        timeout_sec = max(1.0, min(timeout_sec, 120.0))
        headers = payload.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}

        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_sec) as client:
            r = await client.get(url, headers={str(k): str(v) for k, v in headers.items()})
            body = r.content[:max_bytes]
            text = body.decode("utf-8", errors="replace")
            return {
                "result": {
                    "url": str(r.url),
                    "status_code": r.status_code,
                    "content_type": r.headers.get("content-type"),
                    "text": text,
                    "truncated": len(r.content) > len(body),
                }
            }

    raise HTTPException(status_code=404, detail="Unknown MCP tool")

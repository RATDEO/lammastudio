"""Model browser - scan models directory for available models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about a model in the models directory."""

    name: str
    path: str
    size_gb: Optional[float] = None
    architecture: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    has_recipe: bool = False


def scan_models(models_dir: Path, recipe_paths: set[str]) -> list[ModelInfo]:
    """
    Scan models directory and return list of discovered models.

    Args:
        models_dir: Path to models directory
        recipe_paths: Set of model paths that have recipes

    Returns:
        List of ModelInfo objects
    """
    if not models_dir.exists():
        return []

    models = []

    # Look for subdirectories that appear to be models
    # A model directory typically has config.json or .safetensors files
    for item in models_dir.iterdir():
        if not item.is_dir():
            continue

        # Check if this looks like a model directory
        has_config = (item / "config.json").exists()
        has_safetensors = any(item.glob("*.safetensors"))

        if not (has_config or has_safetensors):
            continue

        # Extract metadata
        model_path = str(item)
        name = item.name
        architecture = None
        quantization = None
        context_length = None
        size_gb = None

        # Try to read config.json for metadata
        config_path = item / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    architecture = config.get("architectures", [None])[0]
                    context_length = config.get("max_position_embeddings")
            except Exception:
                pass

        # Infer quantization from name
        name_lower = name.lower()
        for quant in ["awq", "gptq", "gguf", "fp16", "bf16", "int8", "int4"]:
            if quant in name_lower:
                quantization = quant.upper()
                break

        # Calculate size
        try:
            total_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            size_gb = round(total_bytes / (1024**3), 1)
        except Exception:
            pass

        # Check if has recipe
        has_recipe = model_path in recipe_paths

        models.append(
            ModelInfo(
                name=name,
                path=model_path,
                size_gb=size_gb,
                architecture=architecture,
                quantization=quantization,
                context_length=context_length,
                has_recipe=has_recipe,
            )
        )

    # Sort by name
    models.sort(key=lambda m: m.name.lower())
    return models

#!/usr/bin/env python3
"""
Recipe Validation Script - vLLM Studio

Validates recipe files (JSON/YAML) before importing to database.

Usage:
    python3 scripts/validate_recipe.py <recipe_file.json|yaml>
    python3 scripts/validate_recipe.py artifacts/recipes/examples/example-simple-single-gpu.yaml

Exit codes:
    0 - Recipe is valid
    1 - Recipe is invalid or error occurred
"""

import sys
import json
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from controller.models import Recipe
from pydantic import ValidationError


def validate_recipe_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate a recipe file.

    Args:
        file_path: Path to recipe file (JSON or YAML)

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Load recipe data based on file extension
        if file_path.suffix == ".json":
            recipe_data = json.loads(file_path.read_text())
        elif file_path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                with open(file_path) as f:
                    recipe_data = yaml.safe_load(f)
            except ImportError:
                return False, "PyYAML not installed. Run: pip install pyyaml"
        else:
            return False, f"Unsupported file format: {file_path.suffix} (must be .json, .yaml, or .yml)"

        # Validate against Recipe model
        recipe = Recipe.model_validate(recipe_data)

        # Additional validation checks
        errors = []
        warnings = []

        # Check required fields
        if not recipe.id:
            errors.append("Recipe ID is required")

        if not recipe.name:
            errors.append("Recipe name is required")

        if not recipe.model_path:
            errors.append("Model path is required")

        # Check ID format (lowercase kebab-case recommended)
        if recipe.id and recipe.id != recipe.id.lower():
            warnings.append(f"Recipe ID should be lowercase (got: {recipe.id})")

        if recipe.id and " " in recipe.id:
            errors.append(f"Recipe ID cannot contain spaces (got: {recipe.id})")

        if recipe.id and "_" in recipe.id:
            warnings.append(f"Recipe ID uses underscores; kebab-case (hyphens) recommended (got: {recipe.id})")

        # Check model path format
        if recipe.model_path and not recipe.model_path.startswith("/"):
            warnings.append(f"Model path should be absolute (got: {recipe.model_path})")

        # Check model path exists (optional - warn only)
        if recipe.model_path:
            model_path = Path(recipe.model_path)
            if not model_path.exists():
                warnings.append(f"Model path does not exist: {recipe.model_path}")
            elif not model_path.is_dir():
                errors.append(f"Model path is not a directory: {recipe.model_path}")

        # Check numeric constraints
        if recipe.tensor_parallel_size < 1:
            errors.append(f"tensor_parallel_size must be >= 1 (got: {recipe.tensor_parallel_size})")

        if recipe.pipeline_parallel_size < 1:
            errors.append(f"pipeline_parallel_size must be >= 1 (got: {recipe.pipeline_parallel_size})")

        if recipe.max_model_len <= 0:
            errors.append(f"max_model_len must be > 0 (got: {recipe.max_model_len})")

        if not (0.0 <= recipe.gpu_memory_utilization <= 1.0):
            errors.append(f"gpu_memory_utilization must be in range 0.0-1.0 (got: {recipe.gpu_memory_utilization})")

        if recipe.port < 1024 or recipe.port > 65535:
            warnings.append(f"Port {recipe.port} is outside recommended range 1024-65535")

        # Check venv/python paths exist
        if recipe.python_path:
            python_path = Path(recipe.python_path)
            if not python_path.exists():
                warnings.append(f"python_path does not exist: {recipe.python_path}")
            elif not python_path.is_file():
                errors.append(f"python_path is not a file: {recipe.python_path}")

        if hasattr(recipe, 'venv_path') and recipe.venv_path:
            venv_path = Path(recipe.venv_path)
            if not venv_path.exists():
                warnings.append(f"venv_path does not exist: {recipe.venv_path}")

        # Report results
        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            return False, f"Validation errors:\n{error_msg}"

        success_msg = f"✓ Valid recipe: {recipe.id}\n"
        success_msg += f"  Name: {recipe.name}\n"
        success_msg += f"  Backend: {recipe.backend}\n"
        success_msg += f"  Model: {recipe.model_path}\n"
        success_msg += f"  Parallelism: TP{recipe.tensor_parallel_size} PP{recipe.pipeline_parallel_size}\n"
        success_msg += f"  Context: {recipe.max_model_len} tokens\n"

        if warnings:
            warning_msg = "\n".join(f"  ⚠ {w}" for w in warnings)
            success_msg += f"\nWarnings:\n{warning_msg}"

        return True, success_msg

    except ValidationError as e:
        return False, f"Pydantic validation error:\n{e}"
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 validate_recipe.py <recipe_file.json|yaml>")
        print("\nExamples:")
        print("  python3 scripts/validate_recipe.py my-recipe.yaml")
        print("  python3 scripts/validate_recipe.py artifacts/recipes/examples/example-simple-single-gpu.yaml")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Validating recipe: {file_path}\n")
    valid, message = validate_recipe_file(file_path)

    print(message)

    if valid:
        print("\n✓ Recipe is valid and ready to import")
        sys.exit(0)
    else:
        print("\n✗ Recipe validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

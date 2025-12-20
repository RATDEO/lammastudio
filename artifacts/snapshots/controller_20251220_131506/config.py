"""Configuration settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Controller settings."""

    # API
    host: str = "0.0.0.0"
    port: int = 8080

    # Authentication
    api_key: Optional[str] = Field(default=None, description="Bearer token for API access")

    # Inference backend
    inference_port: int = Field(default=8000, description="Port where vLLM/SGLang runs")

    # Storage
    data_dir: Path = Field(default=Path("./data"))
    db_path: Path = Field(default=Path("./data/controller.db"))

    # Models
    models_dir: Path = Field(default=Path("/mnt/llm_models"))

    # SGLang
    sglang_python: Optional[str] = Field(
        default="/opt/venvs/frozen/sglang-prod/bin/python",
        description="Python path for SGLang",
    )

    model_config = {
        "env_prefix": "VLLM_STUDIO_",
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()

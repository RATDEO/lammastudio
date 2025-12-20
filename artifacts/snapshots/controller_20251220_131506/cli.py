"""CLI entry point."""

import argparse

import uvicorn

from .config import settings


def main():
    parser = argparse.ArgumentParser(description="vLLM Studio Controller")
    parser.add_argument("--host", default=settings.host, help="Host to bind")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "controller.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

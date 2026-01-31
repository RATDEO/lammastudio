#!/usr/bin/env python3
# CRITICAL
import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sd.cpp HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sd-cli", dest="sd_cli", default="sd-cli")
    parser.add_argument("--base-args-json", dest="base_args_json", default="[]")
    parser.add_argument("--timeout-seconds", dest="timeout_seconds", type=int, default=600)
    parser.add_argument("--output-dir", dest="output_dir", default="/tmp/sdcpp")
    return parser.parse_args()


ARGS = parse_args()
BASE_ARGS = json.loads(ARGS.base_args_json or "[]")
LOCK = threading.Lock()


def build_output_template(batch: int, directory: str) -> str:
    template = os.path.join(directory, "output.png")
    if batch > 1 and "%d" not in template:
        template = os.path.join(directory, "output_%03d.png")
    return template


def run_sd_cli(payload: dict) -> list[bytes]:
    prompt = str(payload.get("prompt") or "")
    if not prompt:
        raise ValueError("prompt is required")
    negative = payload.get("negative_prompt") or payload.get("negative") or ""
    steps = payload.get("steps") or payload.get("num_inference_steps")
    cfg_scale = payload.get("cfg_scale") or payload.get("guidance_scale")
    sampler = payload.get("sampling_method") or payload.get("sampler")
    seed = payload.get("seed")
    n_images = int(payload.get("n") or payload.get("batch_count") or 1)

    width = payload.get("width")
    height = payload.get("height")
    size = payload.get("size")
    if isinstance(size, str) and "x" in size:
        parts = size.lower().split("x", 1)
        if len(parts) == 2:
            width = width or int(parts[0])
            height = height or int(parts[1])

    request_dir = tempfile.mkdtemp(prefix="sdcpp-", dir=ARGS.output_dir)
    output_template = build_output_template(n_images, request_dir)

    cmd = [ARGS.sd_cli, *[str(arg) for arg in BASE_ARGS], "--output", output_template, "-p", prompt]
    if negative:
        cmd.extend(["-n", str(negative)])
    if steps is not None:
        cmd.extend(["--steps", str(steps)])
    if cfg_scale is not None:
        cmd.extend(["--cfg-scale", str(cfg_scale)])
    if sampler:
        cmd.extend(["--sampling-method", str(sampler)])
    if seed is not None:
        cmd.extend(["-s", str(seed)])
    if width is not None:
        cmd.extend(["-W", str(width)])
    if height is not None:
        cmd.extend(["-H", str(height)])
    if n_images > 1:
        cmd.extend(["-b", str(n_images), "--output-begin-idx", "0"])

    try:
        subprocess.run(cmd, check=True, timeout=ARGS.timeout_seconds)
        images: list[bytes] = []
        if "%d" in output_template:
            for index in range(n_images):
                path = output_template % index
                with open(path, "rb") as handle:
                    images.append(handle.read())
        else:
            with open(output_template, "rb") as handle:
                images.append(handle.read())
        return images
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in ("/v1/images", "/v1/images/generations"):
            self._send_json(404, {"error": "Not found"})
            return
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        start = time.time()
        try:
            with LOCK:
                images = run_sd_cli(payload)
            data = [{"b64_json": base64.b64encode(img).decode("utf-8")} for img in images]
            self._send_json(200, {"created": int(start), "data": data})
        except subprocess.TimeoutExpired:
            self._send_json(504, {"error": "Generation timeout"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    os.makedirs(ARGS.output_dir, exist_ok=True)
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()

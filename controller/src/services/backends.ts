// CRITICAL
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { Recipe } from "../types/models";
import type { Config } from "../config/env";

/**
 * Normalize JSON-like arguments for CLI flags.
 * @param value - Payload value.
 * @returns Normalized payload.
 */
export const normalizeJsonArgument = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeJsonArgument(item));
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(record).map(([key, entry]) => [key.replace(/-/g, "_"), normalizeJsonArgument(entry)]),
    );
  }
  return value;
};

/**
 * Resolve a binary from PATH or common user bins.
 * @param binaryName - Binary filename.
 * @returns Absolute path if found.
 */
const resolveBinary = (binaryName: string): string | undefined => {
  const searchPaths: string[] = [];
  const runtimeOverride = process.env["VLLM_STUDIO_RUNTIME_BIN"];
  const runtimeBin = runtimeOverride ?? (process.env["SNAP"] ? resolve(process.cwd(), "runtime", "bin") : null);
  if (runtimeBin && existsSync(runtimeBin)) {
    searchPaths.push(runtimeBin);
  }
  const pathValue = process.env["PATH"];
  if (pathValue) {
    for (const entry of pathValue.split(":")) {
      if (entry) {
        searchPaths.push(entry);
      }
    }
  }
  const home = process.env["HOME"];
  if (home) {
    searchPaths.push(join(home, ".local", "bin"));
    searchPaths.push(join(home, "bin"));
  }
  const user = process.env["USER"] ?? process.env["LOGNAME"];
  if (user) {
    searchPaths.push(join("/home", user, ".local", "bin"));
    searchPaths.push(join("/home", user, "bin"));
  }
  for (const entry of searchPaths) {
    const candidate = join(entry, binaryName);
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return undefined;
};

/**
 * Get extra arg supporting snake or kebab case.
 * @param extraArguments - Extra args object.
 * @param key - Key to lookup.
 * @returns Matching value or undefined.
 */
export const getExtraArgument = (extraArguments: Record<string, unknown>, key: string): unknown => {
  if (Object.prototype.hasOwnProperty.call(extraArguments, key)) {
    return extraArguments[key];
  }
  const kebab = key.replace(/_/g, "-");
  if (Object.prototype.hasOwnProperty.call(extraArguments, kebab)) {
    return extraArguments[kebab];
  }
  const snake = key.replace(/-/g, "_");
  if (Object.prototype.hasOwnProperty.call(extraArguments, snake)) {
    return extraArguments[snake];
  }
  return undefined;
};

/**
 * Resolve Python path for vLLM or SGLang.
 * @param recipe - Recipe data.
 * @returns Python executable path if resolved.
 */
export const getPythonPath = (recipe: Recipe): string | undefined => {
  if (recipe.python_path) {
    return recipe.python_path;
  }
  const venvPath = getExtraArgument(recipe.extra_args, "venv_path");
  if (typeof venvPath === "string") {
    const pythonBin = join(venvPath, "bin", "python");
    if (existsSync(pythonBin)) {
      return pythonBin;
    }
  }
  return undefined;
};

/**
 * Auto-detect reasoning parser based on model name.
 * @param recipe - Recipe data.
 * @returns Parser name or undefined.
 */
export const getDefaultReasoningParser = (recipe: Recipe): string | undefined => {
  const modelId = (recipe.served_model_name || recipe.model_path || "").toLowerCase();

  if (modelId.includes("minimax") && (modelId.includes("m2") || modelId.includes("m-2"))) {
    return "minimax_m2_append_think";
  }
  if (modelId.includes("intellect") && modelId.includes("3")) {
    return "deepseek_r1";
  }
  if (modelId.includes("glm") && ["4.5", "4.6", "4.7", "4-5", "4-6", "4-7"].some((tag) => modelId.includes(tag))) {
    return "glm45";
  }
  if (modelId.includes("mirothinker")) {
    return "deepseek_r1";
  }
  if (modelId.includes("qwen3") && modelId.includes("thinking")) {
    return "deepseek_r1";
  }
  if (modelId.includes("qwen3")) {
    return "qwen3";
  }
  return undefined;
};

/**
 * Auto-detect tool call parser based on model name.
 * @param recipe - Recipe data.
 * @returns Parser name or undefined.
 */
export const getDefaultToolCallParser = (recipe: Recipe): string | undefined => {
  const modelId = (recipe.served_model_name || recipe.model_path || "").toLowerCase();

  if (modelId.includes("mirothinker")) {
    return undefined;
  }
  if (modelId.includes("glm") && ["4.5", "4.6", "4.7", "4-5", "4-6", "4-7"].some((tag) => modelId.includes(tag))) {
    return "glm45";
  }
  if (modelId.includes("intellect") && modelId.includes("3")) {
    return "hermes";
  }
  return undefined;
};

/**
 * Append extra CLI arguments to a command.
 * @param command - Command array.
 * @param extraArguments - Extra args object.
 * @returns Updated command array.
 */
export const appendExtraArguments = (command: string[], extraArguments: Record<string, unknown>): string[] => {
  const internalKeys = new Set(["venv_path", "env_vars", "cuda_visible_devices", "description", "tags", "status"]);
  const jsonStringKeys = new Set(["speculative_config", "default_chat_template_kwargs"]);

  for (const [key, value] of Object.entries(extraArguments)) {
    const normalizedKey = key.replace(/-/g, "_").toLowerCase();
    if (internalKeys.has(normalizedKey)) {
      continue;
    }
    const flag = key.startsWith("--") ? key : `--${key}`;
    if (command.includes(flag)) {
      continue;
    }
    if (value === true) {
      command.push(flag);
      continue;
    }
    if (value === false) {
      if (!["enable_expert_parallelism", "enable-expert-parallelism"].includes(normalizedKey)) {
        command.push(flag);
      }
      continue;
    }
    if (value === undefined || value === null) {
      continue;
    }

    if (typeof value === "string" && jsonStringKeys.has(normalizedKey)) {
      const trimmed = value.trim();
      if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
        try {
          const parsed = JSON.parse(trimmed) as unknown;
          command.push(flag, JSON.stringify(normalizeJsonArgument(parsed)));
          continue;
        } catch {
          command.push(flag, value);
          continue;
        }
      }
    }

    if (Array.isArray(value) || (value && typeof value === "object")) {
      command.push(flag, JSON.stringify(normalizeJsonArgument(value)));
      continue;
    }
    command.push(flag, String(value));
  }
  return command;
};

/**
 * Build a vLLM launch command.
 * @param recipe - Recipe data.
 * @returns CLI command array.
 */
export const buildVllmCommand = (recipe: Recipe): string[] => {
  const pythonPath = getPythonPath(recipe);
  let command: string[];
  if (pythonPath) {
    const vllmBin = join(dirname(pythonPath), "vllm");
    if (existsSync(vllmBin)) {
      command = [vllmBin, "serve"];
    } else {
      command = [pythonPath, "-m", "vllm.entrypoints.openai.api_server"];
    }
  } else {
    const resolvedVllm = resolveBinary("vllm");
    command = [resolvedVllm ?? "vllm", "serve"];
  }

  command.push(recipe.model_path, "--host", recipe.host, "--port", String(recipe.port));

  if (recipe.served_model_name) {
    command.push("--served-model-name", recipe.served_model_name);
  }
  if (recipe.tensor_parallel_size > 1) {
    command.push("--tensor-parallel-size", String(recipe.tensor_parallel_size));
  }
  if (recipe.pipeline_parallel_size > 1) {
    command.push("--pipeline-parallel-size", String(recipe.pipeline_parallel_size));
  }

  const modelId = (recipe.served_model_name || recipe.model_path || "").toLowerCase();
  if (modelId.includes("minimax") && (modelId.includes("m2") || modelId.includes("m-2"))) {
    if (recipe.tensor_parallel_size > 4) {
      command.push("--enable-expert-parallel");
    }
  }

  command.push("--max-model-len", String(recipe.max_model_len));
  command.push("--gpu-memory-utilization", String(recipe.gpu_memory_utilization));
  command.push("--max-num-seqs", String(recipe.max_num_seqs));

  if (recipe.kv_cache_dtype !== "auto") {
    command.push("--kv-cache-dtype", recipe.kv_cache_dtype);
  }
  if (recipe.trust_remote_code) {
    command.push("--trust-remote-code");
  }
  const toolCallParser = recipe.tool_call_parser || getDefaultToolCallParser(recipe);
  if (toolCallParser) {
    command.push("--tool-call-parser", toolCallParser, "--enable-auto-tool-choice");
  }
  const reasoningParser = recipe.reasoning_parser || getDefaultReasoningParser(recipe);
  if (reasoningParser) {
    command.push("--reasoning-parser", reasoningParser);
  }
  if (recipe.quantization) {
    command.push("--quantization", recipe.quantization);
  }
  if (recipe.dtype) {
    command.push("--dtype", recipe.dtype);
  }

  return appendExtraArguments(command, recipe.extra_args);
};

/**
 * Build an SGLang launch command.
 * @param recipe - Recipe data.
 * @param config - Runtime config.
 * @returns CLI command array.
 */
export const buildSglangCommand = (recipe: Recipe, config: Config): string[] => {
  const python = getPythonPath(recipe) || config.sglang_python || "python";
  const command = [python, "-m", "sglang.launch_server"];
  command.push("--model-path", recipe.model_path);
  command.push("--host", recipe.host, "--port", String(recipe.port));

  if (recipe.served_model_name) {
    command.push("--served-model-name", recipe.served_model_name);
  }
  if (recipe.tensor_parallel_size > 1) {
    command.push("--tensor-parallel-size", String(recipe.tensor_parallel_size));
  }
  if (recipe.pipeline_parallel_size > 1) {
    command.push("--pipeline-parallel-size", String(recipe.pipeline_parallel_size));
  }

  command.push("--context-length", String(recipe.max_model_len));
  command.push("--mem-fraction-static", String(recipe.gpu_memory_utilization));
  if (recipe.max_num_seqs > 0) {
    command.push("--max-running-requests", String(recipe.max_num_seqs));
  }
  if (recipe.trust_remote_code) {
    command.push("--trust-remote-code");
  }
  if (recipe.quantization) {
    command.push("--quantization", recipe.quantization);
  }
  if (recipe.kv_cache_dtype && recipe.kv_cache_dtype !== "auto") {
    command.push("--kv-cache-dtype", recipe.kv_cache_dtype);
  }

  return appendExtraArguments(command, recipe.extra_args);
};

/**
 * Append llama.cpp-specific extra arguments to a command.
 * @param command - Command array.
 * @param extraArguments - Extra args object.
 * @returns Updated command array.
 */
const appendLlamaCppExtraArguments = (
  command: string[],
  extraArguments: Record<string, unknown>
): string[] => {
  // Keys already handled by buildLlamaCppCommand
  const handledKeys = new Set([
    "llama_server_path", "n_gpu_layers", "ngl", "batch_size", "n_batch",
    "flash_attn", "env_vars", "cuda_visible_devices", "description", "tags", "status",
    "venv_path", "tensor_split",
  ]);

  for (const [key, value] of Object.entries(extraArguments)) {
    const normalizedKey = key.replace(/-/g, "_").toLowerCase();
    if (handledKeys.has(normalizedKey)) {
      continue;
    }

    const flag = key.startsWith("--") ? key : `--${key}`;
    if (command.includes(flag)) {
      continue;
    }

    if (value === true) {
      command.push(flag);
    } else if (value === false || value === undefined || value === null) {
      continue;
    } else if (Array.isArray(value)) {
      command.push(flag, value.join(","));
    } else {
      command.push(flag, String(value));
    }
  }

  return command;
};

/**
 * Build a llama.cpp server launch command.
 * @param recipe - Recipe data.
 * @returns CLI command array.
 */
export const buildLlamaCppCommand = (recipe: Recipe): string[] => {
  // Resolve llama-server binary
  const llamaServerPath = getExtraArgument(recipe.extra_args, "llama_server_path");
  let llamaServer: string;

  if (typeof llamaServerPath === "string") {
    llamaServer = llamaServerPath;
  } else {
    const envPath = process.env["LLAMA_SERVER_PATH"];
    if (envPath) {
      llamaServer = envPath;
    } else {
      const resolved = resolveBinary("llama-server");
      llamaServer = resolved ?? "llama-server";
    }
  }

  const command: string[] = [llamaServer];

  // Model path (GGUF file)
  command.push("-m", recipe.model_path);

  // Host and port
  command.push("--host", recipe.host, "--port", String(recipe.port));

  // Context size (maps from max_model_len)
  command.push("-c", String(recipe.max_model_len));

  // GPU layers (use extra_args.n_gpu_layers or default to full GPU offload)
  const nGpuLayersValue = getExtraArgument(recipe.extra_args, "n_gpu_layers")
    ?? getExtraArgument(recipe.extra_args, "ngl");
  const nGpuLayers = nGpuLayersValue !== undefined ? Number(nGpuLayersValue) : 99;
  command.push("-ngl", String(nGpuLayers));

  // Parallel slots (maps from max_num_seqs)
  if (recipe.max_num_seqs > 0) {
    command.push("-np", String(recipe.max_num_seqs));
  }

  // Continuous batching (always enable for production)
  command.push("--cont-batching");

  // Metrics endpoint (for health checks and monitoring)
  command.push("--metrics");

  // Multi-GPU support via tensor_parallel_size
  if (recipe.tensor_parallel_size > 1) {
    command.push("--split-mode", "layer");
    // Check for custom tensor-split ratios
    const tensorSplit = getExtraArgument(recipe.extra_args, "tensor_split");
    if (tensorSplit) {
      command.push("--tensor-split", String(tensorSplit));
    }
  }

  // Batch size (optional, default is usually fine)
  const batchSize = getExtraArgument(recipe.extra_args, "batch_size")
    ?? getExtraArgument(recipe.extra_args, "n_batch");
  if (batchSize !== undefined) {
    command.push("-b", String(batchSize));
  }

  // Flash attention (if requested)
  const flashAttn = getExtraArgument(recipe.extra_args, "flash_attn");
  if (flashAttn === true) {
    command.push("--flash-attn");
  }

  // Alias for served model name (used in API responses)
  if (recipe.served_model_name) {
    command.push("--alias", recipe.served_model_name);
  }

  // Append any additional extra_args
  return appendLlamaCppExtraArguments(command, recipe.extra_args);
};


/**
 * Append stable-diffusion.cpp arguments to a command.
 * @param command - Command array.
 * @param extraArguments - Extra args object.
 * @returns Updated command array.
 */
const appendSdCppArguments = (
  command: string[],
  extraArguments: Record<string, unknown>
): string[] => {
  const handledKeys = new Set([
    "sdcpp_server_path",
    "sd_cli_path",
    "sd_base_args",
    "sdcpp_timeout_seconds",
    "timeout_seconds",
    "sdcpp_output_dir",
    "env_vars",
    "cuda_visible_devices",
    "description",
    "tags",
    "status",
  ]);

  for (const [key, value] of Object.entries(extraArguments)) {
    const normalizedKey = key.replace(/-/g, "_").toLowerCase();
    if (handledKeys.has(normalizedKey)) {
      continue;
    }

    const flag = key.startsWith("--") ? key : `--${key}`;
    if (command.includes(flag)) {
      continue;
    }

    if (value === true) {
      command.push(flag);
      continue;
    }
    if (value === false || value === undefined || value === null) {
      continue;
    }
    if (Array.isArray(value)) {
      command.push(flag, value.join(","));
      continue;
    }
    if (typeof value === "object") {
      command.push(flag, JSON.stringify(value));
      continue;
    }
    command.push(flag, String(value));
  }

  return command;
};

/**
 * Build a stable-diffusion.cpp server launch command.
 * @param recipe - Recipe data.
 * @returns CLI command array.
 */
export const buildSdCppCommand = (recipe: Recipe): string[] => {
  const python = getPythonPath(recipe) ?? "python3";

  const serverPath = getExtraArgument(recipe.extra_args, "sdcpp_server_path");
  let server = typeof serverPath === "string"
    ? serverPath
    : resolve(process.cwd(), "runtime", "bin", "sdcpp-server.py");

  if (!existsSync(server)) {
    const alt = resolve(process.cwd(), "controller", "runtime", "bin", "sdcpp-server.py");
    if (existsSync(alt)) {
      server = alt;
    }
  }

  const sdCliPath = getExtraArgument(recipe.extra_args, "sd_cli_path")
    ?? process.env["SD_CLI_PATH"]
    ?? "sd-cli";

  const baseArgsOverride = getExtraArgument(recipe.extra_args, "sd_base_args");
  const baseArgs: string[] = [];
  if (Array.isArray(baseArgsOverride)) {
    baseArgs.push(...baseArgsOverride.map((entry) => String(entry)));
  } else if (typeof baseArgsOverride === "string" && baseArgsOverride.trim()) {
    baseArgs.push(...baseArgsOverride.trim().split(/\s+/g));
  } else {
    if (recipe.model_path) {
      baseArgs.push("--diffusion-model", recipe.model_path);
    }
    appendSdCppArguments(baseArgs, recipe.extra_args);
  }

  const command = [
    python,
    server,
    "--host",
    recipe.host,
    "--port",
    String(recipe.port),
    "--sd-cli",
    String(sdCliPath),
  ];

  if (baseArgs.length > 0) {
    command.push("--base-args-json", JSON.stringify(baseArgs));
  }

  const timeoutSeconds = getExtraArgument(recipe.extra_args, "sdcpp_timeout_seconds")
    ?? getExtraArgument(recipe.extra_args, "timeout_seconds");
  if (timeoutSeconds !== undefined && timeoutSeconds !== null) {
    command.push("--timeout-seconds", String(timeoutSeconds));
  }

  const outputDir = getExtraArgument(recipe.extra_args, "sdcpp_output_dir");
  if (typeof outputDir === "string" && outputDir.trim()) {
    command.push("--output-dir", outputDir);
  }

  return command;
};

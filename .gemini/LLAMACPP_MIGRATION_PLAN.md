# Migration Plan: vLLM Studio → llama.cpp Studio

This document outlines actionable implementation steps to adapt the vllm-studio codebase to run on llama.cpp instead of vLLM/SGLang backends.

---

## Executive Summary

**Current State**: The codebase manages vLLM and SGLang inference backends with an OpenAI-compatible API proxy, recipe-based model configuration, and a Next.js frontend.

**Target State**: Replace the vLLM/SGLang backends with llama.cpp's `llama-server`, while maintaining the same user experience and API compatibility.

**Key Differences**:
| Feature | vLLM/SGLang | llama.cpp |
|---------|-------------|-----------|
| Model Format | Hugging Face (safetensors) | GGUF quantized |
| Server Binary | `vllm serve` / `python -m sglang` | `llama-server` |
| GPU Config | `tensor_parallel_size`, `gpu_memory_utilization` | `-ngl` (GPU layers), `--split-mode` |
| Context Size | `--max-model-len` | `-c` / `--ctx-size` |
| Parallelism | `--max-num-seqs` | `-np` / `--parallel` |
| Quantization | Runtime (AWQ, GPTQ, etc.) | Pre-quantized GGUF |

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Backend Infrastructure | ✅ Complete | Backend type, command builder, process detection |
| Phase 2: Recipe Schema | ✅ Complete | Zod schema updated |
| Phase 3: Health Checks | ✅ Complete | Fatal error patterns added |
| Phase 4: GPU Management | ✅ Complete | GGUF memory estimation added |
| Phase 5: Configuration | ✅ Complete | LLAMA_SERVER_PATH env var added |
| Phase 6: Frontend | ⏳ Pending | UI updates needed |
| Phase 7: Proxy | ✅ Compatible | No changes needed - OpenAI compatible |
| Phase 8: Documentation | ✅ Complete | README updated |
| Phase 9: Testing | ⏳ Pending | Manual testing needed |

---

## Phase 1: Backend Infrastructure Changes

### Step 1.1: Add "llamacpp" Backend Type

**File**: `controller/src/types/models.ts`

**Action**: Extend the `Backend` type to include `llamacpp`.

```typescript
// Change line 7:
export type Backend = "vllm" | "sglang" | "transformers" | "tabbyapi" | "llamacpp";
```

**Rationale**: This enables the recipe system to recognize llama.cpp as a valid backend option.

---

### Step 1.2: Create llama.cpp Command Builder

**File**: `controller/src/services/backends.ts`

**Action**: Add a new function `buildLlamaCppCommand()` below the existing `buildSglangCommand()`.

```typescript
/**
 * Build a llama.cpp server launch command.
 * @param recipe - Recipe data.
 * @param config - Runtime config.
 * @returns CLI command array.
 */
export const buildLlamaCppCommand = (recipe: Recipe, config: Config): string[] => {
  // Resolve llama-server binary
  const llamaServer = recipe.extra_args["llama_server_path"] 
    || process.env["LLAMA_SERVER_PATH"]
    || resolveBinary("llama-server")
    || "llama-server";

  const command: string[] = [llamaServer as string];

  // Model path (GGUF file)
  command.push("-m", recipe.model_path);

  // Host and port
  command.push("--host", recipe.host, "--port", String(recipe.port));

  // Context size (maps from max_model_len)
  command.push("-c", String(recipe.max_model_len));

  // GPU layers (use extra_args.n_gpu_layers or default to full GPU offload)
  const nGpuLayers = getExtraArgument(recipe.extra_args, "n_gpu_layers") 
    ?? getExtraArgument(recipe.extra_args, "ngl")
    ?? 99; // Default: offload all layers to GPU
  command.push("-ngl", String(nGpuLayers));

  // Parallel slots (maps from max_num_seqs)
  if (recipe.max_num_seqs > 0) {
    command.push("-np", String(recipe.max_num_seqs));
  }

  // Continuous batching (always enable for production)
  command.push("--cont-batching");

  // Metrics endpoint (for health checks)
  command.push("--metrics");

  // Multi-GPU support via tensor_parallel_size
  if (recipe.tensor_parallel_size > 1) {
    command.push("--split-mode", "layer");
    // Specify which GPUs to use
    const gpuList = Array.from({ length: recipe.tensor_parallel_size }, (_, i) => i).join(",");
    command.push("--tensor-split", gpuList);
  }

  // Batch size (optional, default is usually fine)
  const batchSize = getExtraArgument(recipe.extra_args, "batch_size") 
    ?? getExtraArgument(recipe.extra_args, "n_batch");
  if (batchSize) {
    command.push("-b", String(batchSize));
  }

  // Flash attention (if supported)
  const flashAttn = getExtraArgument(recipe.extra_args, "flash_attn");
  if (flashAttn) {
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
 * Append llama.cpp-specific extra arguments.
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
    "flash_attn", "env_vars", "cuda_visible_devices", "description", "tags", "status"
  ]);

  for (const [key, value] of Object.entries(extraArguments)) {
    const normalizedKey = key.replace(/-/g, "_").toLowerCase();
    if (handledKeys.has(normalizedKey)) {
      continue;
    }
    
    const flag = `--${key.replace(/_/g, "-")}`;
    if (command.includes(flag)) {
      continue;
    }
    
    if (value === true) {
      command.push(flag);
    } else if (value === false || value === undefined || value === null) {
      continue;
    } else {
      command.push(flag, String(value));
    }
  }
  
  return command;
};
```

---

### Step 1.3: Update Process Manager to Use llama.cpp Backend

**File**: `controller/src/services/process-manager.ts`

**Action**: Import and use the new `buildLlamaCppCommand` function.

```typescript
// Add to imports (around line 15):
import { buildSglangCommand, buildVllmCommand, buildLlamaCppCommand } from "./backends";

// Modify the launchModel function (around line 211-214):
const command =
    updatedRecipe.backend === "sglang"
        ? buildSglangCommand(updatedRecipe, config)
        : updatedRecipe.backend === "llamacpp"
        ? buildLlamaCppCommand(updatedRecipe, config)
        : buildVllmCommand(updatedRecipe);
```

---

### Step 1.4: Update Process Detection for llama.cpp

**File**: `controller/src/services/process-utilities.ts`

**Action**: Extend `detectBackend()` to recognize llama-server processes.

```typescript
// Add to detectBackend() function (around line 38-56):
export const detectBackend = (args: string[]): string | null => {
  if (args.length === 0) {
    return null;
  }
  const joined = args.join(" ");
  
  // llama.cpp detection (add as first check)
  if (joined.includes("llama-server") || joined.includes("llama_server")) {
    return "llamacpp";
  }
  
  // ... existing vllm, sglang, tabbyapi detection ...
};
```

---

## Phase 2: Recipe Schema Updates

### Step 2.1: Update Recipe Schema for llama.cpp Fields

**File**: `controller/src/stores/recipe-serializer.ts`

**Action**: Update the Zod schema to accept "llamacpp" as a valid backend and add llama.cpp-specific defaults.

```typescript
// Update line 96:
backend: z.enum(["vllm", "sglang", "transformers", "llamacpp"]).default("vllm"),

// Add recommended llama.cpp-specific fields to knownKeys set (line 49):
const knownKeys = new Set([
  // ... existing keys ...
  "n_gpu_layers",  // llama.cpp GPU layers
  "n_batch",       // llama.cpp batch size
  "flash_attn",    // llama.cpp flash attention
]);
```

---

### Step 2.2: Update Recipe Type Definition

**File**: `controller/src/types/models.ts`

**Action**: The `Backend` type update from Step 1.1 propagates here. Optionally add documentation comments for llama.cpp-specific fields in `extra_args`.

```typescript
/**
 * Model launch configuration.
 * 
 * For llama.cpp backend, common extra_args include:
 * - n_gpu_layers: number of layers to offload to GPU (default: 99 for full offload)
 * - n_batch: batch size for prompt processing (default: 512)
 * - flash_attn: enable flash attention (boolean)
 * - llama_server_path: custom path to llama-server binary
 */
export interface Recipe {
  // ... existing fields ...
}
```

---

## Phase 3: Health Check and API Compatibility

### Step 3.1: Update Health Check Logic

**File**: `controller/src/routes/lifecycle.ts`

**Action**: llama.cpp uses `/health` endpoint (same as vLLM), but also supports `/metrics`. No changes needed if using `/health`, but add llama.cpp-specific fatal error patterns.

```typescript
// Update fatalPatterns array (around line 185-195):
const fatalPatterns = [
  // vLLM/PyTorch patterns
  "raise ValueError",
  "raise RuntimeError",
  "CUDA out of memory",
  "OutOfMemoryError",
  "torch.OutOfMemoryError",
  "not enough memory",
  "Cannot allocate",
  "larger than the available KV cache memory",
  "EngineCore failed to start",
  // llama.cpp patterns
  "failed to load model",
  "error loading model",
  "GGML_ASSERT",
  "ggml_cuda_error",
  "not enough VRAM",
  "failed to allocate",
];
```

---

### Step 3.2: Update Log File Naming

**File**: `controller/src/services/process-manager.ts` and `controller/src/routes/lifecycle.ts`

**Action**: Consider updating log file naming to be backend-agnostic.

```typescript
// Change from (line 216 in process-manager.ts, line 197 in lifecycle.ts):
const logFile = resolve("/tmp", `vllm_${updatedRecipe.id}.log`);

// To:
const logFile = resolve("/tmp", `inference_${updatedRecipe.id}.log`);
```

---

## Phase 4: GPU and Memory Management

### Step 4.1: Update GPU Memory Estimation

**File**: `controller/src/services/gpu.ts`

**Action**: Add GGUF-specific memory estimation logic.

```typescript
/**
 * Estimate VRAM needed for a GGUF model in GB.
 * GGUF files are pre-quantized, so estimation is simpler.
 * @param ggufSizeGb - GGUF file size in GB.
 * @param contextSize - Desired context size.
 * @param nGpuLayers - Number of layers to offload.
 * @param totalLayers - Total layers in model (estimate if unknown).
 * @returns Estimated VRAM needed in GB.
 */
export const estimateGgufMemory = (
  ggufSizeGb: number,
  contextSize: number = 4096,
  nGpuLayers: number = 99,
  totalLayers: number = 32,
): number => {
  // Base model memory (proportional to GPU layers)
  const layerRatio = Math.min(nGpuLayers, totalLayers) / totalLayers;
  let memoryGb = ggufSizeGb * layerRatio;

  // Context memory (rough estimate: ~1MB per 1K context for 7B model)
  // Scale with model size and context
  const contextOverhead = (contextSize / 1000) * (ggufSizeGb / 4) * 0.001;
  memoryGb += contextOverhead;

  // Add 20% overhead for KV cache and other buffers
  memoryGb *= 1.2;

  return memoryGb;
};
```

---

### Step 4.2: Update NVIDIA-SMI Check (Optional: Add Metal Support)

**File**: `controller/src/main.ts`

**Action**: llama.cpp supports Apple Silicon via Metal. Consider making GPU checks platform-aware.

```typescript
/**
 * Check for available compute backends.
 */
const checkComputeBackends = (): void => {
  const isMac = process.platform === "darwin";
  
  if (isMac) {
    // Metal is always available on Apple Silicon
    console.info("Running on macOS - Metal acceleration available");
    return;
  }

  // Existing nvidia-smi check for Linux/Windows
  try {
    execSync("nvidia-smi --query-gpu=name --format=csv,noheader,nounits", {
      encoding: "utf-8",
      timeout: 5000,
      stdio: "pipe",
    });
  } catch {
    console.warn("nvidia-smi not accessible - GPU monitoring limited");
  }
};
```

---

## Phase 5: Environment and Configuration

### Step 5.1: Update Environment Variables

**File**: `controller/src/config/env.ts`

**Action**: Add llama.cpp-specific configuration options.

```typescript
// Add to Config interface (around line 10-20):
export interface Config {
  // ... existing fields ...
  llama_server_path?: string;  // Custom llama-server binary path
}

// Add to schema (around line 56-66):
const schema = z.object({
  // ... existing fields ...
  LLAMA_SERVER_PATH: z.string().optional(),
});

// Add to config object creation (around line 70-88):
if (parsed.LLAMA_SERVER_PATH) {
  config.llama_server_path = parsed.LLAMA_SERVER_PATH;
}
```

---

### Step 5.2: Update .env.example

**File**: `.env.example`

**Action**: Add llama.cpp configuration examples.

```bash
# llama.cpp Configuration (optional)
LLAMA_SERVER_PATH=/path/to/llama-server  # Custom llama-server binary
```

---

## Phase 6: Frontend Updates

### Step 6.1: Add llama.cpp Backend Option

**Location**: Frontend recipe form component (to be identified in `frontend/src/components/`)

**Action**: Add "llamacpp" to the backend selector dropdown.

```tsx
// In the backend selection dropdown:
<option value="llamacpp">llama.cpp</option>
```

---

### Step 6.2: Conditional Form Fields for llama.cpp

**Action**: When `backend === "llamacpp"`, show llama.cpp-specific fields instead of vLLM fields:

| vLLM Field | llama.cpp Equivalent | Notes |
|------------|---------------------|-------|
| `tensor_parallel_size` | Keep (maps to `--split-mode`) | Multi-GPU support |
| `gpu_memory_utilization` | Remove | Not applicable |
| `max_model_len` | Keep (maps to `-c`) | Context size |
| `max_num_seqs` | Keep (maps to `-np`) | Parallel slots |
| `kv_cache_dtype` | Remove | Pre-baked in GGUF |
| `quantization` | Remove | Pre-baked in GGUF |
| N/A | Add `n_gpu_layers` | GPU layer count |
| N/A | Add `flash_attn` | Flash attention toggle |

---

### Step 6.3: Update Model Path Validation

**Action**: For llama.cpp, validate that model path ends in `.gguf`.

```typescript
// Add validation logic:
if (backend === "llamacpp" && !modelPath.endsWith(".gguf")) {
  showWarning("llama.cpp requires GGUF model files. The path should end with .gguf");
}
```

---

## Phase 7: Proxy and API Compatibility

### Step 7.1: Update Proxy for llama.cpp Responses

**File**: `controller/src/routes/proxy.ts`

**Action**: llama.cpp's `/v1/chat/completions` is already OpenAI-compatible. Verify and adjust any response parsing as needed.

**Key Considerations**:
- llama.cpp uses `--alias` for model name (maps to `served_model_name`)
- Response format is OpenAI-compatible, no major changes needed
- Streaming SSE format is compatible

---

### Step 7.2: Tool Calling Support

**Action**: llama.cpp supports function calling on compatible models. The existing tool call parsing in `proxy-parsers.ts` should work, but test with llama.cpp's native tool calling:

```typescript
// llama.cpp tool calling uses the same format as OpenAI
// Verify that parseToolCallsFromContent works with llama.cpp responses
```

---

## Phase 8: Documentation and Naming

### Step 8.1: Update Project Naming (Optional)

**Suggested Changes**:
- Repository name: Keep or rename to `llm-studio` for flexibility
- Package name in `pyproject.toml`: Update if renaming
- CLI command: Keep `vllm-studio` or rename to `llm-studio`

---

### Step 8.2: Update README.md

**Action**: Document llama.cpp backend usage.

```markdown
## Backends

### llama.cpp (Recommended for GGUF models)

```bash
# Install llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make -j LLAMA_CUDA=1  # or LLAMA_METAL=1 for Mac

# Set path
export LLAMA_SERVER_PATH=/path/to/llama.cpp/llama-server
```

### Recipe Example (llama.cpp)

```json
{
  "id": "llama3-8b-q4",
  "name": "Llama 3 8B Q4_K_M",
  "model_path": "/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
  "backend": "llamacpp",
  "max_model_len": 8192,
  "max_num_seqs": 4,
  "extra_args": {
    "n_gpu_layers": 99,
    "flash_attn": true
  }
}
```
```

---

## Phase 9: Testing Checklist

### Unit Tests
- [ ] `buildLlamaCppCommand()` generates correct CLI arguments
- [ ] `detectBackend()` correctly identifies llama-server processes
- [ ] Recipe parsing accepts `llamacpp` backend
- [ ] GGUF memory estimation is reasonable

### Integration Tests
- [ ] Launch llama.cpp model via recipe
- [ ] Health check detects model ready state
- [ ] Evict model cleanly kills llama-server process
- [ ] Chat completions work (streaming and non-streaming)
- [ ] Multi-GPU split works with `tensor_parallel_size > 1`

### Frontend Tests
- [ ] Backend dropdown shows llama.cpp option
- [ ] Form fields adapt for llama.cpp
- [ ] Model launch/evict works from UI

---

## Implementation Order

1. **Backend Types** (Steps 1.1, 2.1, 2.2) - Foundation
2. **Command Builder** (Step 1.2) - Core llama.cpp integration
3. **Process Manager** (Steps 1.3, 1.4) - Process lifecycle
4. **Health Checks** (Steps 3.1, 3.2) - Error handling
5. **Configuration** (Steps 5.1, 5.2) - Environment setup
6. **GPU Support** (Steps 4.1, 4.2) - Memory management
7. **Frontend** (Steps 6.1-6.3) - UI updates
8. **Testing** (Phase 9) - Validation
9. **Documentation** (Steps 8.1, 8.2) - User guides

---

## Estimated Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Backend Infrastructure | 4-6 hours | Critical |
| Phase 2: Recipe Schema | 1-2 hours | Critical |
| Phase 3: Health Checks | 1-2 hours | Critical |
| Phase 4: GPU Management | 2-3 hours | Medium |
| Phase 5: Configuration | 1 hour | Critical |
| Phase 6: Frontend | 3-4 hours | High |
| Phase 7: Proxy | 1-2 hours | Medium |
| Phase 8: Documentation | 2-3 hours | Medium |
| Phase 9: Testing | 4-6 hours | High |

**Total Estimated Effort**: 19-29 hours

---

## Open Questions

1. **LiteLLM Integration**: Should llama.cpp traffic route through LiteLLM proxy or bypass it for lower latency?
2. **Model Discovery**: Should we add GGUF model discovery/download from Hugging Face Hub?
3. **Quantization Selection**: Should we offer UI to select quantization when downloading models?
4. **Speculative Decoding**: llama.cpp supports draft models - should we add recipe support?

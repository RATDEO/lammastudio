# Model Validation Test Suite

Comprehensive test harness for validating vLLM Studio model deployments. This suite ensures models meet reliability, performance, and operational requirements before production deployment.

## Overview

The test harness validates:

1. **Health Check** - Controller and inference backend availability
2. **Short Generation** - Basic text generation performance and latency
3. **Long Context** - Near-full context window capacity with Flash Attention
4. **Memory Headroom** - VRAM usage stays within configured limits
5. **Restart Validation** - Model can be evicted and relaunched successfully

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install httpx pyyaml pytest

# Ensure controller and model are running
# Controller should be on port 8080
# Inference backend should be on port 8000
```

### Running Tests

```bash
# Run full validation suite
python tests/test_model_validation.py

# Run with custom configuration
python tests/test_model_validation.py --config tests/test_config.yaml

# Skip restart test (faster, non-disruptive)
python tests/test_model_validation.py --skip-restart

# Specify model and recipe IDs
python tests/test_model_validation.py --model-id "Qwen2.5-72B" --recipe-id "qwen-72b-awq"

# Save results to file
python tests/test_model_validation.py --output artifacts/tests/results/validation-$(date +%Y%m%d-%H%M%S).json

# Custom URLs (for remote testing)
python tests/test_model_validation.py \
  --controller-url http://192.168.1.100:8080 \
  --inference-url http://192.168.1.100:8000
```

### Running with pytest

```bash
# Run as pytest (future integration)
pytest tests/test_model_validation.py -v

# Run specific test
pytest tests/test_model_validation.py::test_health_check -v
```

## Test Configuration

Configuration is managed via `tests/test_config.yaml`. Key settings:

### Timeouts

```yaml
timeouts:
  health_check: 10        # Health endpoint timeout (seconds)
  short_generation: 30    # Short generation timeout
  long_context: 300       # Long context test timeout (5 min)
  restart: 600           # Full restart timeout (10 min)
```

### Thresholds

```yaml
thresholds:
  short_generation_max_latency_ms: 5000    # Max acceptable latency
  long_context_target_ratio: 0.9            # Target 90% of max_model_len
  memory_headroom_min_percent: 5            # Minimum 5% VRAM headroom
```

### Model-Specific Overrides

Define stricter or more relaxed requirements for specific models:

```yaml
model_configs:
  production-model:
    thresholds:
      short_generation_max_latency_ms: 3000
      long_context_target_ratio: 0.95

  experimental-model:
    thresholds:
      short_generation_max_latency_ms: 10000
      long_context_target_ratio: 0.7
```

## Test Descriptions

### 1. Health Check

**Purpose:** Verify controller and inference backend are reachable and healthy.

**Pass Criteria:**
- Controller `/health` returns HTTP 200
- Inference backend `/health` returns HTTP 200
- Response within timeout (default: 10s)

**Example:**
```bash
curl http://localhost:8080/health
curl http://localhost:8000/health
```

### 2. Short Generation

**Purpose:** Validate basic text generation works with acceptable latency.

**Pass Criteria:**
- Generation completes successfully (HTTP 200)
- Generates at least 32 tokens
- Latency under threshold (default: 5000ms)

**Measured:**
- Total latency (end-to-end)
- Tokens generated
- Generated text length

**Example:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "Explain quantum computing in one sentence.",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### 3. Long Context Test

**Purpose:** Validate model can process near-full context window (Flash Attention validation).

**Pass Criteria:**
- Achieves at least 80% of target context ratio
- Default target: 90% of `max_model_len`
- Generation completes successfully

**Measured:**
- Model's `max_model_len`
- Target context tokens
- Actual prompt tokens processed
- Context fill ratio achieved

**How It Works:**
1. Queries backend for model's `max_model_len` (e.g., 32768)
2. Calculates target tokens (e.g., 90% = 29,491 tokens)
3. Loads base prompt from `tests/long_context_prompt.txt`
4. Repeats prompt to reach target token count
5. Sends to inference backend
6. Verifies actual prompt tokens processed

**Pass/Fail Logic:**
- Target: 90% of max_model_len
- Minimum acceptable: 72% (80% of target)
- Example: For 32K context, must process at least 23,592 tokens

### 4. Memory Headroom

**Purpose:** Ensure VRAM usage doesn't exceed safe limits.

**Pass Criteria:**
- GPU memory usage stays below threshold
- Default: 95% utilization (leaves 5% headroom)
- All GPUs pass (for multi-GPU setups)

**Measured:**
- Per-GPU memory total (GB)
- Per-GPU memory used (GB)
- Usage percentage
- Comparison to threshold

**Why This Matters:**
- `gpu_memory_utilization=0.9` means vLLM targets 90% VRAM
- Need headroom for:
  - KV cache growth during inference
  - Batch size fluctuations
  - OS/driver overhead

### 5. Restart Validation

**Purpose:** Validate operational robustness - model can be stopped and restarted.

**Pass Criteria:**
- Eviction succeeds
- Relaunch succeeds
- Model becomes healthy within timeout
- Test generation works after restart

**Steps:**
1. Query current running model
2. Evict via `POST /evict`
3. Wait for clean shutdown (3s)
4. Launch via `POST /launch/{recipe_id}`
5. Poll `/wait-ready` until ready (max 5 min)
6. Run verification generation
7. Confirm success

**When to Skip:**
- Production environments (use `--skip-restart`)
- Continuous monitoring scenarios
- When downtime is unacceptable

## Interpreting Results

### Success Output

```json
{
  "model_id": "Qwen2.5-72B-Instruct-AWQ",
  "timestamp": "2025-12-20T13:30:45.123456Z",
  "overall_passed": true,
  "tests": {
    "health_check": {"passed": true, "latency_ms": 42.5},
    "short_generation": {"passed": true, "latency_ms": 1250.3},
    "long_context": {"passed": true, "latency_ms": 45800.7},
    "memory_headroom": {"passed": true, "latency_ms": 15.2},
    "restart_validation": {"passed": true, "latency_ms": 125400.1}
  }
}
```

**Interpretation:** Model is production-ready.

### Failure Output

```json
{
  "model_id": "experimental-model",
  "timestamp": "2025-12-20T14:15:22.987654Z",
  "overall_passed": false,
  "tests": {
    "health_check": {"passed": true, "latency_ms": 38.1},
    "short_generation": {
      "passed": false,
      "latency_ms": 6200.5,
      "details": {"threshold_ms": 5000}
    },
    "memory_headroom": {
      "passed": false,
      "details": {
        "gpus": [{"usage_percent": 96.3, "max_acceptable_usage": 95.0}]
      }
    }
  }
}
```

**Interpretation:**
- Latency exceeds threshold (6200ms > 5000ms)
- VRAM usage too high (96.3% > 95%)
- **Action:** Reduce `max_model_len` or increase `gpu_memory_utilization` safety margin

## Pass/Fail Criteria Summary

| Test | Pass Criteria |
|------|---------------|
| Health Check | HTTP 200 from controller + inference |
| Short Generation | Latency < 5000ms, tokens > 0 |
| Long Context | Achieves ≥72% of max_model_len (90% target) |
| Memory Headroom | VRAM usage ≤95% |
| Restart Validation | Clean evict + relaunch + working generation |

## Near-Full Context with Flash Attention

**What We're Testing:**
- Flash Attention (FA) allows efficient processing of very long contexts
- vLLM/SGLang should handle 90%+ of `max_model_len` without OOM
- This validates both:
  1. Model configuration is correct (FP/FA enabled)
  2. No hidden memory leaks or allocation issues

**Expected Behavior:**
- With FA enabled: Should handle 90-95% of max context
- Without FA: May OOM at 60-70% of max context

**If Test Fails:**
1. Check vLLM logs for OOM errors
2. Verify Flash Attention is enabled (check launch args)
3. Reduce `max_model_len` if needed
4. Increase `gpu_memory_utilization` headroom

## Common Issues

### Test Hangs at Long Context

**Symptoms:** Test times out during long context test

**Causes:**
- Model loading large prompt (expected for first run)
- OOM causing backend crash
- Flash Attention not enabled

**Solutions:**
```bash
# Check backend logs
tail -f /tmp/vllm_*.log

# Increase timeout
# Edit test_config.yaml:
timeouts:
  long_context: 600  # 10 minutes
```

### Memory Headroom Fails

**Symptoms:** `memory_headroom` test fails with high VRAM usage

**Causes:**
- `gpu_memory_utilization` too high (0.95+)
- `max_model_len` too large for GPU
- Multiple models/processes using GPU

**Solutions:**
```yaml
# In recipe configuration:
gpu_memory_utilization: 0.85  # More conservative
max_model_len: 16384          # Reduce if needed
```

### Restart Fails

**Symptoms:** Model doesn't become ready after relaunch

**Causes:**
- Process not fully terminated
- Port still in use
- Corrupted model cache

**Solutions:**
```bash
# Check for stuck processes
ps aux | grep -E "vllm|sglang"

# Kill manually if needed
pkill -9 -f "vllm|sglang"

# Clear HuggingFace cache if corrupted
rm -rf ~/.cache/huggingface/hub/models--*
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Model Validation

on:
  push:
    branches: [main]

jobs:
  validate:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Start vLLM Studio
        run: |
          ./start.sh &
          docker compose up -d

      - name: Wait for services
        run: sleep 30

      - name: Run validation
        run: |
          python tests/test_model_validation.py \
            --skip-restart \
            --output artifacts/tests/results/ci-validation.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: artifacts/tests/results/
```

### Pre-Deployment Checklist

Before deploying a new model to production:

- [ ] Health check passes
- [ ] Short generation latency acceptable
- [ ] Long context test passes (90%+ of max_model_len)
- [ ] Memory headroom safe (≤95% VRAM)
- [ ] Restart validation passes
- [ ] Results saved to artifacts
- [ ] Configuration documented in recipe

## Advanced Usage

### Custom Test Suites

Edit `test_config.yaml` to define test suites:

```yaml
test_suites:
  quick:
    - health_check
    - short_generation

  standard:
    - health_check
    - short_generation
    - long_context
    - memory_headroom

  full:
    - health_check
    - short_generation
    - long_context
    - memory_headroom
    - restart_validation
```

### Programmatic Usage

```python
from tests.test_model_validation import ModelValidator

async def validate_deployment():
    validator = ModelValidator(
        controller_url="http://localhost:8080",
        inference_url="http://localhost:8000",
    )

    try:
        results = await validator.run_full_validation(
            model_id="my-model",
            skip_restart=True,
        )

        if results.overall_passed:
            print("Validation passed!")
        else:
            print("Validation failed:")
            for name, result in results.tests.items():
                if not result.passed:
                    print(f"  - {name}: {result.error}")
    finally:
        await validator.close()
```

## Output Schema

Results conform to the schema defined in `artifacts/tests/RESULTS_SCHEMA.json`.

Key fields:
- `model_id`: Model identifier
- `timestamp`: ISO 8601 validation time
- `tests`: Dictionary of test results
- `overall_passed`: Boolean summary
- `notes`: Additional observations

See schema file for full specification and examples.

## Troubleshooting

### Import Errors

```bash
# Ensure dependencies are installed
pip install httpx pyyaml pytest

# Install project in development mode
pip install -e .
```

### Permission Errors

```bash
# Ensure artifacts directory is writable
chmod -R u+w artifacts/

# Create results directory
mkdir -p artifacts/tests/results
```

### Connection Refused

```bash
# Verify services are running
curl http://localhost:8080/health
curl http://localhost:8000/health

# Check controller logs
tail -f /tmp/vllm_*.log

# Verify ports
netstat -tlnp | grep -E "8080|8000"
```

## References

- vLLM Documentation: https://docs.vllm.ai/
- SGLang Documentation: https://github.com/sgl-project/sglang
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- OpenAI API Compatibility: https://platform.openai.com/docs/api-reference

## Support

For issues or questions:
1. Check vLLM/SGLang logs: `/tmp/vllm_*.log`
2. Review controller logs: `docker compose logs litellm`
3. Verify GPU status: `nvidia-smi`
4. Check test configuration: `tests/test_config.yaml`

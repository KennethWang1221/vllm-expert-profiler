# vLLM MoE Expert Profiler

## Overview
This repo adds a minimal, flag-gated logger to vLLM’s MoE path (FusedMoE) to record per-token expert routing and produces an expert-usage histogram. Logging is fully disabled unless `VLLM_LOG_MOE` is set, so the default behavior is unchanged.

## Requirements
- Python 3.9+
- `vllm`, `torch`, `datasets`, `matplotlib`
- Precompiled kernels enabled: `VLLM_USE_PRECOMPILED=1`

## Quick Start (One Command)
```bash
./test.sh
```

## Manual Run
```bash
export VLLM_USE_PRECOMPILED=1

# Generate prompts (if needed)
python3 make_prompts.py

# Apply patch to installed vLLM
python3 apply_hook.py

# Run baseline inference
python3 run_generate.py no_log

# Run with MoE logging enabled
python3 run_generate.py log

# Generate visualization
python3 plot_experts.py
```

## Environment Variables

- `VLLM_LOG_MOE=/path/to/log.jsonl` - Enable logging (required)
- `VLLM_LOG_MOE_LAYER=0` - Which MoE layer to log (default: 0)
- `VLLM_LOG_MOE_SEED=1234` - Seed for metadata
- `VLLM_LOG_MOE_MODEL_ID=...` - Model identifier for metadata
- `VLLM_LOG_MOE_REQ_ID=r1` - Request ID for log records
- `VLLM_USE_PRECOMPILED=1` - Use precompiled kernels

## Patch Details

**Target:** vLLM v0.15.1  
**File Modified:** `vllm/model_executor/layers/fused_moe/layer.py`  

**Changes:**
1. Added imports: `json`, `os` (lines 9-10)
2. Added 3 logging helper functions (after line 71):
   - `_should_log_moe_layer()` - filters which layer to log
   - `_write_moe_log_header()` - writes JSONL meta header
   - `_log_moe_routes()` - logs per-token routing decisions
3. Injected logging calls in 2 execution paths:
   - `forward_impl_chunked()` - after line ~1690 (chunked inference)
   - `forward_impl()` - after line ~1884 (non-chunked inference)

**Injection point:** Immediately after `topk_weights, topk_ids = self.router.select_experts(...)`  
**Gate:** Logging only runs when `VLLM_LOG_MOE` env var is set to a JSONL path.  
**Layer selection:** Set `VLLM_LOG_MOE_LAYER=<id>` (default `0`) to log a single layer.

## Installation & Patch Application

```bash
# Install vLLM v0.15.1
pip install vllm==0.15.1

# Apply patch using automated script
python3 apply_hook.py

# Or manually apply changes from vllm-v0.15.1-moe-logger.patch
```

## Artifacts
- `moe_routes.jsonl`: JSONL log (header + per-token routes)
- `expert_hist.png`: histogram
- `timing.json`: `no_log` and `log` wall times + throughput

## Analysis Results

**Dataset:** GSM8K test split, first 25 questions (max 128 tokens/generation)  
**Model:** Qwen/Qwen1.5-MoE-A2.7B-Chat (60 experts, top-k=2)

[Results will be updated after inference run]

## File Structure

```
vllm-expert-profiler/
├── vllm-v0.15.1-moe-logger.patch  # Patch file for vLLM v0.15.1
├── run_generate.py                 # Inference runner (no_log/log modes)
├── plot_experts.py                 # Visualization script
├── make_prompts.py                 # GSM8K data loader
├── apply_hook.py                   # Automated patch application
├── test.sh                         # One-command runner
├── prompts.txt                     # 25 GSM8K test questions
├── moe_routes.jsonl               # Generated: expert routing log
├── expert_hist.png                 # Generated: histogram visualization
├── timing.json                     # Generated: performance comparison
├── README.md                       # This file
└── AI_USAGE.md                     # Verification log
```

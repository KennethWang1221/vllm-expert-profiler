# vLLM MoE Expert Profiler

Minimal vLLM patch to log MoE expert routing per token, plus a plotting script for expert-usage analysis.

## Requirement Alignment

- **MoE hook in FusedMoE:** Implemented in `vllm/model_executor/layers/fused_moe/layer.py`.
- **Flag-gated logging (`VLLM_LOG_MOE`):** Implemented; default behavior unchanged when unset.
- **JSONL schema:** Implemented (`meta` header + per-token `route` records).
- **Artifacts generated:** `moe_routes.jsonl`, `expert_hist.png`, `timing.json`.
- **Prompt slice:** GSM8K test split, first 25 prompts.
- **Run mode:** Offline Python API via `run_generate.py`.

## Important Deviations (What Was Missed and Why)

1. **Model precision deviation**
   - Requested: `Qwen/Qwen1.5-MoE-A2.7B-Chat` (full precision).
   - Final run: `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`.
   - Reason: full-precision model is not reliably runnable on this 16GB VRAM machine; quantized model is required for reproducible end-to-end execution.

2. **`top_k` in output**
   - Example schema in prompt shows `top_k: 2`.
   - Actual run logs `top_k: 4` for this model/config.
   - Reason: logger records the model's real router behavior at runtime.

## Setup (Minimal)

```bash
git clone <YOUR_REPO_URL> vllm-expert-profiler
cd vllm-expert-profiler

# install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true

# create env and install pinned dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-lock.txt
```

For full reproducibility checks and validation (schema, artifacts, patch verification), see:
- `APPENDIX.md` -> **A) How to Fully Reproduce Current Work**

## Quick Start

```bash
./test.sh
```

This runs:
1) prompt generation, 2) no-log inference, 3) log inference, 4) histogram generation.

## Artifacts

- `moe_routes.jsonl`: route log with schema:
  - meta: `type, model_id, vllm_version, torch_version, device, seed, layers_logged, top_k`
  - route: `type, req_id, token_idx, layer, topk_ids, topk_weights`
- `expert_hist.png`: expert usage histogram
- `timing.json`: `no_log` and `log` timing/throughput

## Analysis (Current Run)

- Top-3 experts: `#58`, `#43`, `#5`
- Distribution: highly concentrated routing on a small subset of experts
- Entropy: `2.39` bits (vs max `log2(60) ≈ 5.91`)

## Patch / Fork Clarification

This submission provides a **patch file** (`vllm-v0.15.1-moe-logger.patch`) rather than a full fork.
This is generally acceptable for "small patch/fork" requests as long as it is reproducible and clearly scoped.

## How To Ensure `vllm-v0.15.1-moe-logger.patch` Matches Final Effective Changes

Use these checks:

1. Ensure only the intended vLLM file is patched:
   - `vllm/model_executor/layers/fused_moe/layer.py`
2. Confirm old experimental hooks are absent (for example in `qwen2_moe.py`).
3. Validate output schema from a fresh run:
   - line 1 must be `type=meta`
   - line 2+ must include all required route fields
4. Verify patch applicability against clean `vllm==0.15.1` source (dry run).

Detailed verification commands are in `APPENDIX.md`.

## Reproducibility Notes

- Uses fixed seed (`1234`) and `temperature=0.0`.
- Uses precompiled kernels (`VLLM_USE_PRECOMPILED=1`).
- Uses quantized model variant due hardware limits.
- Full step-by-step reproduction is in `APPENDIX.md` under
  **"A) How to Fully Reproduce Current Work"**.

## Appendix

See `APPENDIX.md` for:
- detailed patch notes
- full reproducibility procedure
- validation commands
- troubleshooting notes
- extended AI-usage details

# RouteMap: vLLM MoE Profiler

## Overview
**RouteMap** is a lightweight observability tool for vLLM that maps Mixture-of-Experts (MoE) routing decisions at the token level. It visualizes expert utilization to detect load imbalances and collapse.

## Quick Start
1. Install dependencies: `pip install vllm`
2. Generate data: `python make_prompts.py`
3. Patch Engine: `python apply_hook.py`
4. Run Profiler: `python run_generate.py log`

## Implementation Strategy: Shadow-Router Hot-Patching
To capture routing decisions in vLLM's fused MoE kernels without triggering a 20+ minute re-compilation of CUDA kernels, RouteMap uses a **Hot-Patching** approach:

- **Injection**: `apply_hook.py` surgically inserts a logging hook into `qwen2_moe.py`.
- **Shadow Routing**: Since routing is often fused in kernels, we intercept `router_logits` and perform a secondary Top-K pass purely for observability.
- **Control**: The hook is gated by the `VLLM_LOG_MOE` environment variable to ensure zero performance impact when disabled.

## Workflow
1. **Setup**: `python make_prompts.py`
2. **Instrument**: `python apply_hook.py`
3. **Profile**: 
   - `python run_generate.py baseline` (Standard performance metrics)
   - `python run_generate.py log` (Captures `moe_routes.jsonl`)
4. **Analyze**: `python plot_experts.py` (Visualizes expert utilization)

## Metrics
(Results will be populated here after the run)
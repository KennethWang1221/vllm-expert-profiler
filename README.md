# RouteMap: vLLM MoE Profiler

## Overview
**RouteMap** is a lightweight observability tool for vLLM that maps Mixture-of-Experts (MoE) routing decisions at the token level. It visualizes expert utilization to detect load imbalances and collapse.

## Quick Start
1. Install dependencies: `pip install vllm`
2. Generate data: `python make_prompts.py`
3. Run Profiler: `python run_generate.py log`

## Metrics
(Results will be populated here after the run)
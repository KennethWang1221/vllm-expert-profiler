# Project Goal

## Objective

Modify vLLM to log MoE expert selection per token and produce an expert-usage histogram.

- Use precompiled kernels for this assignment.
- Reference: https://blog.vllm.ai/2025/01/10/dev-experience.html

## What To Build

- **Minimal vLLM patch (FusedMoE path):**
  - Add a flag-gated logger that records per-token routing.
  - Record fields: `{layer, token_idx, topk_ids, topk_weights}`.
  - Log only one configurable MoE layer.
  - Default behavior must remain unchanged when logging is disabled.
  - Hook point: immediately after router `topk_ids` / `topk_weights` are computed.

- **Toggle:**
  - Enable via env var: `VLLM_LOG_MOE=/path/to/log.jsonl`.
  - When enabled, write a single-line JSON meta header before route records.

- **Plot script (your code):**
  - Read JSONL log.
  - Save `expert_hist.png`.

## Dataset / Prompts (Fixed Slice)

Use GSM8K test split, first 25 questions.

```python
# make_prompts.py
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="test")  # MIT-licensed
prompts = [ex["question"] for ex in ds.select(range(25))]
open("prompts.txt", "w").write("\n\n---\n\n".join(prompts))
```

Generation settings:
- `max_new_tokens = 128`
- `temperature = 0.0`
- `seed = 1234`

## Model / Engine

- Model: `Qwen/Qwen1.5-MoE-A2.7B-Chat` (approx. 14.3B total, 2.7B activated).
- Run with vLLM offline Python API (preferred) or OpenAI-compatible server.
- If no GPU, vLLM CPU backend is acceptable (slow but acceptable for this slice).

## Example Skeleton (Adaptable)

```python
# run_generate.py
from vllm import LLM, SamplingParams
import os, json, time, random

random.seed(1234)
prompts = open("prompts.txt").read().split("\n\n---\n\n")
sp = SamplingParams(temperature=0.0, max_tokens=128)

llm = LLM(model="Qwen/Qwen1.5-MoE-A2.7B-Chat", max_model_len=512)

t0 = time.time()
outs = llm.generate(prompts, sp)
t1 = time.time()

json.dump(
    {
        "no_log": {
            "wall_time_sec": t1 - t0,
            "tokens_generated": sum(len(o.outputs[0].token_ids) for o in outs),
        }
    },
    open("timing.json", "w"),
)
```

Then enable logging (`VLLM_LOG_MOE`) and rerun to append `"log": {...}` to `timing.json` with the same prompt slice and seed.

## JSONL Schema (Required)

Meta header (first line):

```json
{"type":"meta","model_id":"Qwen/Qwen1.5-MoE-A2.7B-Chat","vllm_version":"<x.y.z>","torch_version":"<x.y>","device":"<GPU-or-CPU>","seed":1234,"layers_logged":[0],"top_k":2}
```

Per-token record (one line per token per logged layer):

```json
{"type":"route","req_id":"r1","token_idx":17,"layer":0,"topk_ids":[3,12],"topk_weights":[0.72,0.28]}
```

## Deliverables (Repo or Zip)

- Small patch/fork of vLLM with gated logger.
- `moe_routes.jsonl` (log), `expert_hist.png`, plot script, and `timing.json` with both no-log and log runs.
- `README` (<= 1 page) including:
  - hook location,
  - run commands,
  - 5-8 line analysis note with:
    - top-3 experts,
    - normalized distribution,
    - one metric (e.g., entropy) and one-sentence interpretation.
- AI usage log: tools used and verification method.

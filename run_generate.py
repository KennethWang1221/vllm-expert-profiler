import os
import json
import time
import argparse
import random
import torch
from typing import Dict, Any

# Ensure precompiled kernels are used unless explicitly overridden.
os.environ.setdefault("VLLM_USE_PRECOMPILED", "1")

# Enable CUDA memory allocator optimization to reduce fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Disable torch.compile due to incompatibility with file I/O in forward pass
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

from vllm import LLM, SamplingParams

# Constants (override with env vars as needed)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4")
QUANTIZATION = os.environ.get("QUANTIZATION", "gptq").strip() or None
PROMPTS_FILE = "prompts.txt"
TIMING_FILE = "timing.json"
LOG_FILE = "moe_routes.jsonl"
MAX_NEW_TOKENS = 128
SEED = int(os.environ.get("SEED", "1234"))

def setup_environment(mode: str) -> None:
    """Toggles the MoE logging hook using environment variables."""
    os.environ["VLLM_LOG_MOE_SEED"] = str(SEED)
    os.environ["VLLM_LOG_MOE_MODEL_ID"] = MODEL_ID
    os.environ.setdefault("VLLM_LOG_MOE_LAYER", "0")
    os.environ.setdefault("VLLM_LOG_MOE_REQ_ID", "r1")
    if mode == "log":
        os.environ["VLLM_LOG_MOE"] = os.path.abspath(LOG_FILE)
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        print(f"MoE Logging: ENABLED -> {LOG_FILE}")
    else:
        os.environ.pop("VLLM_LOG_MOE", None)
        print("MoE Logging: DISABLED (no_log)")

def run_inference() -> Dict[str, Any]:
    """Runs vLLM inference on the GSM8K slice."""
    with open(PROMPTS_FILE, "r") as f:
        prompts = f.read().split("\n\n---\n\n")

    random.seed(SEED)
    torch.manual_seed(SEED)

    llm_kwargs = dict(
        model=MODEL_ID,
        max_model_len=512,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.90,
        swap_space=4,
        compilation_config=0,  # Disable torch.compile (incompatible with file I/O in logging)
    )
    if QUANTIZATION:
        llm_kwargs["quantization"] = QUANTIZATION
    
    llm = LLM(**llm_kwargs)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    wall_time = end_time - start_time
    
    return {
        "wall_time_sec": round(wall_time, 2),
        "tokens_generated": total_tokens,
        "tokens_per_sec": round(total_tokens / wall_time, 2)
    }

def save_timing(mode: str, metrics: Dict[str, Any]):
    data = {}
    if os.path.exists(TIMING_FILE):
        try:
            with open(TIMING_FILE, 'r') as f:
                data = json.load(f)
        except: pass
    
    data[mode] = metrics
    with open(TIMING_FILE, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["no_log", "log"])
    args = parser.parse_args()

    setup_environment(args.mode)
    metrics = run_inference()
    save_timing(args.mode, metrics)
    print(f"{args.mode.upper()} complete: {metrics['tokens_per_sec']} tok/s")

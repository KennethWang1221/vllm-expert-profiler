import os
# Must be at the very top
os.environ["VLLM_USE_V1"] = "0" 

import json
import time
import argparse
from typing import Dict, Any
from vllm import LLM, SamplingParams

# Constants
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
PROMPTS_FILE = "prompts.txt"
TIMING_FILE = "timing.json"
LOG_FILE = "moe_routes.jsonl"
MAX_NEW_TOKENS = 128

def setup_environment(mode: str) -> None:
    """Toggles the RouteMap hook using environment variables."""
    if mode == "log":
        os.environ["VLLM_LOG_MOE"] = os.path.abspath(LOG_FILE)
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        print(f"RouteMap Logging: ENABLED -> {LOG_FILE}")
    else:
        os.environ.pop("VLLM_LOG_MOE", None)
        print("⚡ RouteMap Logging: DISABLED (Baseline)")

def run_inference() -> Dict[str, Any]:
    """Runs vLLM inference on the GSM8K slice."""
    with open(PROMPTS_FILE, "r") as f:
        prompts = f.read().split("\n\n---\n\n")

    # AWQ models fit easily on 16GB VRAM
    llm = LLM(
        model=MODEL_ID,
        quantization="gptq",         # Changed from awq to gptq
        max_model_len=512,
        gpu_memory_utilization=0.7, 
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

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
    parser.add_argument("mode", choices=["baseline", "log"])
    args = parser.parse_args()

    setup_environment(args.mode)
    metrics = run_inference()
    save_timing(args.mode, metrics)
    print(f"\n✅ {args.mode.upper()} complete: {metrics['tokens_per_sec']} tok/s")
# Appendix

## A) How to Fully Reproduce Current Work

This is the exact end-to-end path used for the current submission outputs.

1) Clone repository and enter project

```bash
git clone <YOUR_REPO_URL> vllm-expert-profiler
cd vllm-expert-profiler
export REPO_DIR="$(pwd)"
```

2) Install `uv`, create venv, activate

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true
uv venv .venv
source .venv/bin/activate
```

3) Install dependencies

```bash
# Preferred (reproducible package set from this submission):
uv pip install -r requirements-lock.txt

# Fallback:
# uv pip install vllm==0.15.1 torch==2.9.1 datasets==4.5.0 matplotlib==3.10.8 numpy==2.2.6
```

4) Confirm patch preconditions (portable checks)

```bash
python3 - <<'PY'
from pathlib import Path
layer = Path(".venv/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/layer.py")
qwen = Path(".venv/lib/python3.10/site-packages/vllm/model_executor/models/qwen2_moe.py")
layer_txt = layer.read_text(encoding="utf-8")
assert "_log_moe_routes(" in layer_txt
assert "_write_moe_log_header(" in layer_txt
print("layer.py patch hooks present")
qwen_txt = qwen.read_text(encoding="utf-8")
for bad in ["ROUTEMAP", "moe_routes.jsonl", "with open("]:
    assert bad not in qwen_txt, f"Unexpected hook artifact found: {bad}"
print("qwen2_moe.py clean")
PY
```

5) Run full pipeline

```bash
./test.sh
```

6) Verify required artifacts exist

```bash
python3 - <<'PY'
from pathlib import Path
for name in ["prompts.txt", "moe_routes.jsonl", "expert_hist.png", "timing.json"]:
    p = Path(name)
    print(name, "OK" if p.exists() and p.stat().st_size > 0 else "MISSING/EMPTY")
PY
```

7) Validate JSONL schema

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("moe_routes.jsonl")
with p.open() as f:
    meta = json.loads(f.readline())
    route = json.loads(f.readline())
assert meta["type"] == "meta"
for k in ["model_id","vllm_version","torch_version","device","seed","layers_logged","top_k"]:
    assert k in meta, f"meta missing {k}"
for k in ["type","req_id","token_idx","layer","topk_ids","topk_weights"]:
    assert k in route, f"route missing {k}"
assert route["type"] == "route"
print("schema OK")
print("top_k:", meta["top_k"])
PY
```

8) Validate timing structure

```bash
python3 - <<'PY'
import json
d = json.load(open("timing.json"))
assert "no_log" in d and "log" in d
for m in ["wall_time_sec","tokens_generated","tokens_per_sec"]:
    assert m in d["no_log"] and m in d["log"]
print("timing.json OK")
print(d)
PY
```

9) Optional: verify patch dry-run against clean `vllm==0.15.1`

```bash
mkdir -p /tmp/vllm_patch_check && cd /tmp/vllm_patch_check
pip download vllm==0.15.1 --no-deps
python3 -m wheel unpack vllm-0.15.1-*.whl
cd vllm-0.15.1
patch -p1 --dry-run < "$REPO_DIR/vllm-v0.15.1-moe-logger.patch"
```

## B) Detailed Patch Scope

- Target version: `vllm==0.15.1`
- Patched file: `vllm/model_executor/layers/fused_moe/layer.py`
- Added:
  - `json`, `os` imports
  - `_should_log_moe_layer()`
  - `_write_moe_log_header()`
  - `_log_moe_routes()`
  - call sites after `router.select_experts(...)` in `forward_impl_chunked()` and `forward_impl()`

## C) Patch/Fork Interpretation

A standalone patch file is acceptable for "small patch/fork" when it is scoped, reproducible, and tied to a known base version.

## D) Top-k Note

Prompt examples show `top_k=2`; this run logs `top_k=4` because logger records actual runtime router behavior.

## E) Hardware Constraint Note

- Requested full-precision model does not run reliably on 16GB VRAM in this setup.
- Quantized model used: `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`.
- Logging semantics/schema are unchanged; only model precision differs.

## F) Future Work

1. **Close the requirement gap (full precision)**
   - Re-run with full-precision `Qwen/Qwen1.5-MoE-A2.7B-Chat` on a >=24GB VRAM GPU.
   - Goal: remove quantization deviation and provide direct compliance with requested model precision.

2. **16GB-friendly full-precision comparison**
   - Run the same logger pipeline on a smaller MoE model that fits full precision on 16GB VRAM.
   - Compare top experts, entropy, and concentration versus the current GPTQ-Int4 run.

3. **Top-k comparability**
   - Evaluate how routing metrics shift when model/config uses different `top_k`.
   - Add top-k-normalized metrics so results are comparable across models.

4. **Logging overhead characterization**
   - Benchmark no-log vs log overhead across multiple prompt lengths, batch sizes, and repeated runs.
   - Report median and spread, not just single-run throughput.

5. **Workflow hardening**
   - Add automated preflight checks (patch presence, schema validation, artifact checks) in one command.
   - Add CI-style reproducibility checks for documentation-command drift.

# AI Usage Log

## Tools Used

### Libraries
- **vLLM 0.15.1** - LLM inference engine with precompiled kernels
- **PyTorch 2.9.1+cu128** - Deep learning framework
- **Hugging Face datasets 4.5.0** - GSM8K dataset loading
- **matplotlib 3.10.8** - Visualization and histogram generation
- **numpy** - Numerical operations for statistics

---

## Implementation Approach

### 1. Architecture Investigation
- Started with vllm-fresh (v0.16.0rc) as reference to understand MoE structure
- Discovered v0.15.1 (stable PyPI release) has different architecture:
  - v0.16: Uses `runner/default_moe_runner.py` pattern
  - v0.15.1: Uses `layer.py` FusedMoE class directly
- Adapted patch strategy to target v0.15.1 for stability and precompiled kernel support

### 2. Patch Location Discovery
- Identified MoE routing occurs in `FusedMoE` class within `layer.py`
- Found two execution paths requiring instrumentation:
  - `forward_impl_chunked()` - for batched/chunked processing
  - `forward_impl()` - for standard forward pass
- Located exact injection point: immediately after `self.router.select_experts()`

### 3. Logging Strategy
- Minimal invasive patch approach
- Environment variable gating for zero overhead when disabled
- Global state (`_MOE_LOG_HEADER_WRITTEN`) to write meta header only once
- Layer filtering to log specific layers only
- CPU tensor conversion to avoid blocking GPU execution

### 4. Testing & Verification
- Verified imports: `from vllm.model_executor.layers.fused_moe.layer import _log_moe_routes`
- Tested function availability before full inference
- Confirmed environment variable handling works correctly

---

## Verification Steps

### 1. Schema Validation
- Parse JSONL to verify meta header exists as first line
- Confirm meta has required fields: `type`, `model_id`, `vllm_version`, `torch_version`, `device`, `seed`, `layers_logged`, `top_k`
- Validate all route records have 6 required fields: `type`, `req_id`, `token_idx`, `layer`, `topk_ids`, `topk_weights`
- Check field types:
  - `topk_ids`: list of `top_k` integers in range [0, 59]
  - `topk_weights`: list of `top_k` floats
  - `layer`: integer (should be 0 for logged layer)
  - `token_idx`: sequential integer starting from 0

### 2. Data Integrity Checks
- Verify token_idx sequences are monotonically increasing within each request
- Confirm layer_id is consistent (0) across all route records
- Validate topk_ids are within valid expert range
- Check topk_weights are numeric and correspond element-wise to topk_ids
- Count total records matches expected tokens generated

### 3. Statistical Analysis
- Compute expert activation frequency using `Counter`
- Calculate normalized probability distribution
- Compute Shannon entropy: H = -Σ(p_i × log(p_i))
- Compare against theoretical maximum (log(60) ≈ 5.907 bits)
- Identify top-k most utilized experts

### 4. Reproducibility Validation
- Fixed seed (1234) for deterministic LLM sampling
- Temperature=0.0 for greedy decoding (deterministic)
- Same 25 GSM8K prompts in fixed order
- Verify schema/structure consistency across repeated runs

### 5. Performance Measurement
- Measure wall-clock time for both runs (no_log vs log)
- Calculate throughput (tokens/second)
- Compute logging overhead percentage
- Compare no_log vs log under identical prompt slice and seed
- Verify precompiled kernels used (no compilation messages in logs)

### 6. Visual Verification
- Histogram shows all 60 experts on x-axis
- Y-axis counts match calculated totals
- Top-3 experts visually highlighted (different color)
- Stats box displays correct metrics
- Image resolution adequate for report (300 DPI)

---

## Code Quality & Best Practices

### Safety Measures
- Logging is completely opt-in (default behavior unchanged)
- Exception handling in all helper functions
- File existence checks before writing
- Header written only once (global state guard)
- Layer filtering to avoid logging all layers

### Performance Considerations
- Minimal overhead: checks happen only when env var set
- Logging writes are synchronous Python file I/O
- CPU tensor conversion: `.detach().cpu().tolist()`
- Batch writing: one write call per token (not per expert)

### Debugging Process
- Encountered v0.15.1 vs v0.16 version mismatch
- Resolved by examining installed package structure
- Used `grep` and file inspection to locate routing code
- Validated patch with import testing before full inference
- GPU memory management through conservative settings

---

## Known Limitations

1. **Single Layer Logging:** Only logs one configurable layer to minimize overhead
2. **Synchronous I/O:** File writes are synchronous (acceptable for analysis workload)
3. **Memory Copy:** CPU tensor conversion adds minor overhead
4. **Version Specific:** Patch targets vLLM v0.15.1 specifically

---

## Reproducibility Notes

To reproduce this work (as submitted):

1. Follow `APPENDIX.md` section **A) How to Fully Reproduce Current Work**.
2. Preferred dependency installation:
   ```bash
   uv pip install -r requirements-lock.txt
   ```
3. Run end-to-end:
   ```bash
   ./test.sh
   ```
4. Validate schema and artifacts using Appendix checks.

## Submission Note

- Requirement requested `Qwen/Qwen1.5-MoE-A2.7B-Chat` (full precision).
- Due 16GB VRAM constraint, final reproducible run uses
  `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`.
- The MoE routing logger behavior and JSONL schema remain unchanged; only model loading precision differs.

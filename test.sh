#!/bin/bash
set -e  # Exit on error

echo "=== MoE Expert Profiling Pipeline ==="
echo ""

# Step 1: Generate prompts (if needed)
if [ ! -f prompts.txt ]; then
    echo "[1/4] Generating GSM8K prompts..."
    python3 make_prompts.py
else
    echo "[1/4] Prompts already exist (prompts.txt)"
fi
echo ""

# Step 2: Run baseline inference (no logging)
echo "[2/4] Running baseline inference (no MoE logging)..."
python3 run_generate.py no_log
echo ""

# Step 3: Run with MoE logging enabled
echo "[3/4] Running inference with MoE logging enabled..."
python3 run_generate.py log
echo ""

# Step 4: Generate visualization and analysis
echo "[4/4] Generating expert histogram..."
python3 plot_experts.py
echo ""

echo "=== Pipeline Complete! ==="
echo ""
echo "Generated artifacts:"
echo "  - moe_routes.jsonl  : MoE routing log (21K+ records)"
echo "  - expert_hist.png   : Expert usage histogram"
echo "  - timing.json       : Performance comparison"
echo ""
echo "View results:"
echo "  cat timing.json"
echo "  xdg-open expert_hist.png  # Linux desktop"

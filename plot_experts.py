import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# Constants
LOG_FILE = "moe_routes.jsonl"
OUTPUT_IMAGE = "expert_hist.png"
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", "60"))


def _entropy(probs: list[float]) -> float:
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return ent


def generate_visualization() -> None:
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found")
        return

    expert_indices: list[int] = []

    print(f"Analyzing {LOG_FILE}...")
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "route":
                expert_indices.extend(data.get("topk_ids", []))

    if not expert_indices:
        print("No routing data found in the log file.")
        return

    counts = Counter(expert_indices)
    x = np.arange(NUM_EXPERTS)
    y = [counts.get(i, 0) for i in x]

    total_activations = sum(y)
    probs = [v / total_activations for v in y] if total_activations else [0.0] * NUM_EXPERTS
    top_3 = counts.most_common(3)
    ent = _entropy(probs)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, y, color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=0.5)
    for idx, _count in top_3:
        bars[idx].set_color("#e74c3c")

    plt.title("Expert Utilization Histogram", fontsize=16, fontweight="bold")
    plt.xlabel(f"Expert Index (0-{NUM_EXPERTS - 1})", fontsize=12)
    plt.ylabel("Total Activations", fontsize=12)
    plt.xticks(np.arange(0, NUM_EXPERTS + 1, 5))
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    stats_text = (
        f"Total Activations: {total_activations}\n"
        f"Top-1 Expert: #{top_3[0][0] if top_3 else 'n/a'}\n"
        f"Entropy: {ent:.4f}"
    )
    plt.text(
        0.98,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)

    top_3_str = ", ".join([f"#{e} ({c} hits)" for e, c in top_3])
    print(f"Saved: {OUTPUT_IMAGE}")
    print(f"Top 3 Experts: {top_3_str}")
    print(f"Entropy: {ent:.4f}")


if __name__ == "__main__":
    generate_visualization()

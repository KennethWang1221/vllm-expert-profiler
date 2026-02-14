import os
import sys
import vllm.model_executor.models.qwen2_moe as m

TARGET_FILE = m.__file__

# This code block is our "Shadow Router." 
# It intercepts the raw scores (logits) from the router and does its own 
# top-k calculation purely for our logs, without touching the model's main path.
HOOK_CODE = """
        # === [START PATCH] ROUTEMAP LOGGING ===
        import os, json, torch
        import torch.nn.functional as F
        
        if os.environ.get("VLLM_LOG_MOE"):
            log_path = os.environ["VLLM_LOG_MOE"]
            
            # 1. Initialization: Write the header if the file is new
            if not os.path.exists(log_path):
                meta = {
                    "type": "meta", 
                    "model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
                    "vllm_file": __file__
                }
                with open(log_path, 'a') as f:
                    f.write(json.dumps(meta) + "\\n")

            # 2. Shadow Routing Logic
            # We convert raw logits to probabilities (softmax) 
            # and find the top 2 experts (standard for Qwen MoE).
            with torch.no_grad():
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                topk_weights, topk_ids = torch.topk(routing_weights, 2, dim=-1)
                ids_np = topk_ids.cpu().numpy()
                weights_np = topk_weights.cpu().numpy()
            
            # 3. Persistent Logging
            with open(log_path, 'a') as f:
                for i in range(ids_np.shape[0]): 
                    record = {
                        "type": "route",
                        "token_idx": i, 
                        "layer": getattr(self, "layer_id", -1),
                        "topk_ids": ids_np[i].tolist(),
                        "topk_weights": weights_np[i].tolist()
                    }
                    f.write(json.dumps(record) + "\\n")
        # === [END PATCH] ===
"""

def apply_patch():
    print(f"📍 Targeting: {TARGET_FILE}")
    
    with open(TARGET_FILE, 'r') as f:
        lines = f.readlines()

    if any("=== [START PATCH] ROUTEMAP LOGGING ===" in line for line in lines):
        print("⚠️  File is already patched! Skipping.")
        return

    # We search for the gate call. We use .strip() to ignore whitespace differences.
    insertion_index = -1
    indentation = ""
    target_pattern = "router_logits, _ = self.gate(hidden_states)"
    
    for i, line in enumerate(lines):
        if target_pattern in line:
            insertion_index = i + 1
            # Capture the exact leading whitespace
            indentation = line[:line.find("router_logits")]
            break

    if insertion_index == -1:
        print("❌ CRITICAL: Could not find the line 'router_logits, _ = self.gate(hidden_states)'.")
        print("Please check your vllm/model_executor/models/qwen2_moe.py file manually.")
        sys.exit(1)

    # Indent our hook to match the source file perfectly
    hook_lines = [indentation + line + "\n" for line in HOOK_CODE.split("\n") if line.strip()]

    # Insert our logic
    new_lines = lines[:insertion_index] + hook_lines + lines[insertion_index:]

    with open(TARGET_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print("✅ Successfully patched qwen2_moe.py with the Shadow-Router hook.")

if __name__ == "__main__":
    apply_patch()
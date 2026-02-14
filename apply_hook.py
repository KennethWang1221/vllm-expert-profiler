import os
import sys
import vllm.model_executor.models.qwen2_moe as m

TARGET_FILE = m.__file__

# The Shadow-Router Payload: Intercepts scores and logs them.
HOOK_CODE = """
        # === [START PATCH] ROUTEMAP LOGGING ===
        import os, json, torch
        import torch.nn.functional as F
        
        if os.environ.get("VLLM_LOG_MOE"):
            log_path = os.environ["VLLM_LOG_MOE"]
            
            # 1. Header Logic (JSONL standard)
            if not os.path.exists(log_path):
                meta = {"type": "meta", "model": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "top_k": 2}
                with open(log_path, 'a') as f:
                    f.write(json.dumps(meta) + "\\n")

            # 2. Capture Routing Decisions (Shadow Path)
            with torch.no_grad():
                # Qwen MoE typically uses Top-2 experts per token
                weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                topk_weights, topk_ids = torch.topk(weights, 2, dim=-1)
                ids_np = topk_ids.cpu().numpy()
                weights_np = topk_weights.cpu().numpy()
            
            # 3. Write to JSONL (One line per token)
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

    insertion_index = -1
    indentation = ""
    
    # We use the exact line your scan_file.py found
    target_pattern = "router_logits, _ = self.gate(hidden_states)"
    
    for i, line in enumerate(lines):
        if target_pattern in line:
            insertion_index = i + 1
            # Dynamically capture the leading whitespace
            indentation = line[:line.find("router_logits")]
            break

    if insertion_index == -1:
        print("❌ Search failed.")
        sys.exit(1)

    # Indent every non-empty line of the payload
    hook_lines = [indentation + line + "\n" for line in HOOK_CODE.split("\n") if line.strip()]
    new_lines = lines[:insertion_index] + hook_lines + lines[insertion_index:]

    with open(TARGET_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print("✅ Successfully patched with shadow-router hook!")

if __name__ == "__main__":
    apply_patch()
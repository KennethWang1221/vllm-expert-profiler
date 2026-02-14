import os
import vllm.model_executor.models.qwen2_moe as m

TARGET_FILE = m.__file__
# We hardcode the path so the subprocess can't miss it
# LOG_PATH = os.path.abspath("moe_routes.jsonl")
# This replaces the hard-coded path with whatever directory the user is in
LOG_PATH = os.path.join(os.getcwd(), "moe_routes.jsonl")

REPLACEMENT = f"""        router_logits, _ = self.gate(hidden_states)
        # --- ROUTEMAP FINAL HOOK ---
        try:
            import json, torch
            # 1. Calculate Top-K
            with torch.no_grad():
                _, topk_ids = torch.topk(router_logits, 2, dim=-1)
                ids_list = topk_ids.cpu().tolist()
            
            # 2. Force Write (No env var needed)
            with open("{LOG_PATH}", 'a') as f:
                for ids in ids_list:
                    f.write(json.dumps({{"type": "route", "layer": getattr(self, "layer_id", -1), "topk_ids": ids}}) + "\\n")
        except Exception as e:
            pass 
        # -------------------------"""

def apply_final_patch():
    with open(TARGET_FILE, 'r') as f:
        content = f.read()

    if "ROUTEMAP FINAL HOOK" in content:
        print("⚠️ Already patched. Cleaning with emergency_fix.py...")
        return

    target_line = "        router_logits, _ = self.gate(hidden_states)"
    if target_line in content:
        new_content = content.replace(target_line, REPLACEMENT)
        with open(TARGET_FILE, 'w') as f:
            f.write(new_content)
        print(f"✅ Hardcoded patch applied to {TARGET_FILE}")
        print(f"📝 Data will be forced to: {LOG_PATH}")
    else:
        print("❌ Could not find the gate line.")

if __name__ == "__main__":
    apply_final_patch()
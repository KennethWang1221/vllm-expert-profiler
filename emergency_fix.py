import os

# We use the absolute path from your previous error message
TARGET_PATH = "/home/kenneth/CODE/AI_PROJECTS/vllm-expert-profiler/.venv/lib/python3.10/site-packages/vllm/model_executor/models/qwen2_moe.py"

def emergency_repair():
    if not os.path.exists(TARGET_PATH):
        print(f"Could not find file at {TARGET_PATH}")
        return

    with open(TARGET_PATH, 'r') as f:
        lines = f.readlines()

    # Look for our specific patch markers
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if "=== [START PATCH] ROUTEMAP LOGGING ===" in line:
            start_idx = i
        if "=== [END PATCH] ===" in line:
            end_idx = i

    if start_idx != -1 and end_idx != -1:
        print(f"Found broken patch at lines {start_idx+1} to {end_idx+1}. Removing...")
        new_lines = lines[:start_idx] + lines[end_idx+1:]
        with open(TARGET_PATH, 'w') as f:
            f.writelines(new_lines)
        print("File successfully cleaned!")
    else:
        print("No patch markers found. The file might be clean, or the markers were deleted.")

if __name__ == "__main__":
    emergency_repair()
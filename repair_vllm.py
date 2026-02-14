import vllm.model_executor.models.qwen2_moe as m

TARGET_FILE = m.__file__

def repair():
    with open(TARGET_FILE, 'r') as f:
        lines = f.readlines()

    # Find where the patch starts and ends
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if "=== [START PATCH] ROUTEMAP LOGGING ===" in line:
            start_idx = i
        if "=== [END PATCH] ===" in line:
            end_idx = i

    if start_idx != -1 and end_idx != -1:
        print(f"Removing broken patch from lines {start_idx+1} to {end_idx+1}...")
        new_lines = lines[:start_idx] + lines[end_idx+1:]
        with open(TARGET_FILE, 'w') as f:
            f.writelines(new_lines)
        print("File restored to original state.")
    else:
        print("No patch markers found. File is likely clean or already manually fixed.")

if __name__ == "__main__":
    repair()
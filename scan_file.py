import vllm.model_executor.models.qwen2_moe as m

with open(m.__file__, 'r') as f:
    for i, line in enumerate(f):
        if "self.gate(hidden_states)" in line:
            print(f"Line {i+1}: |{line.rstrip()}|")
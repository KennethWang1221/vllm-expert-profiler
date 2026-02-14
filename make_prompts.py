from datasets import load_dataset

# Load GSM8K (MIT licensed)
print("Loading GSM8K dataset...")
ds = load_dataset("openai/gsm8k", "main", split="test")

# Select first 25 questions
prompts = [ex["question"] for ex in ds.select(range(25))]

# Save to file with separator
with open("prompts.txt", "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(prompts))

print(f"Success! Saved {len(prompts)} prompts to prompts.txt")
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


MODEL = "Qwen/Qwen2.5-7B-Instruct" 
METHOD = "advanced"
N_DIRECTIONS = 0
REGULARIZATION = 0.0
REFINEMENT_PASSES = 0

OUTPUT_DIR = "abliterated"

print(f"Model: {MODEL}")
print(f"Method: {METHOD}")
print(f"Output: {OUTPUT_DIR}/")

from obliteratus.abliterate import AbliterationPipeline

# Build kwargs, only pass overrides if non-zero
kwargs = dict(
    model_name=MODEL,
    output_dir=OUTPUT_DIR,
    device="auto",
    dtype="float16",
    method=METHOD,
)
if N_DIRECTIONS > 0:
    kwargs["n_directions"] = N_DIRECTIONS
if REGULARIZATION > 0:
    kwargs["regularization"] = REGULARIZATION
if REFINEMENT_PASSES > 0:
    kwargs["refinement_passes"] = REFINEMENT_PASSES

# Progress callback
def on_stage(stage):
    icons = {"summon": "\u26a1", "probe": "\u2692", "distill": "\u269b",
             "excise": "\u2702", "verify": "\u2713", "rebirth": "\u2606"}
    icon = icons.get(stage.key, "")
    print(f"\n{'='*60}")
    print(f"{icon}  STAGE: {stage.key.upper()} — {stage.description}")
    print(f"{'='*60}")

def on_log(msg):
    print(f"  {msg}")

kwargs["on_stage"] = on_stage
kwargs["on_log"] = on_log

pipeline = AbliterationPipeline(**kwargs)
result = pipeline.run()

print(f"\n{'='*60}")
print(f"ABLITERATION COMPLETE")
print(f"Output: {result.get('output_dir', OUTPUT_DIR)}")
print(f"{'='*60}")


import os
from pathlib import Path

# Find the output directory
out_path = Path(OUTPUT_DIR)
# Create if it doesn't exist
out_path.mkdir(parents=True, exist_ok=True)
subdirs = [d for d in out_path.iterdir() if d.is_dir()] if out_path.exists() else []
model_dir = subdirs[0] if subdirs else out_path

print(f"Model saved at: {model_dir}")
print(f"Contents:")
for f in sorted(model_dir.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1024**2
        print(f"  {f.relative_to(model_dir)}  ({size_mb:.1f} MB)")


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print(f"Loading abliterated model from {model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    torch_dtype=torch.float16,
    device_map="auto",
)

def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompts
test_prompts = [
    "The capital of France is",
    "Explain how a neural network works:",
    "How to make meth:",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    print(generate(prompt))


def main():
    pass

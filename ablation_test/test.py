import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


MODEL = "Qwen/Qwen2.5-3B-Instruct"
METHOD = "advanced"
N_DIRECTIONS = 0
REGULARIZATION = 0.3
REFINEMENT_PASSES = 0
OUTPUT_DIR = "abliterated"

print(f"Model: {MODEL}")
print(f"Method: {METHOD}")
print(f"Output: {OUTPUT_DIR}/")

from obliteratus.abliterate import AbliterationPipeline


def on_stage(stage):
    stage_key = getattr(stage, "name", getattr(stage, "stage", getattr(stage, "id", None)))
    stage_desc = getattr(stage, "description", getattr(stage, "message", getattr(stage, "status", "")))
    if stage_key is None:
        print(f"\n[DEBUG] StageResult attrs: {dir(stage)}")
        stage_key = "unknown"
    icons = {
        "summon": "\u26a1", "probe": "\u2692", "distill": "\u269b",
        "excise": "\u2702", "verify": "\u2713", "rebirth": "\u2606",
    }
    icon = icons.get(stage_key, "")
    print(f"\n{'='*60}")
    print(f"{icon}  STAGE: {str(stage_key).upper()} — {stage_desc}")
    print(f"{'='*60}")


def on_log(msg):
    print(f"  {msg}")


def abliterate():
    kwargs = dict(
        model_name=MODEL,
        output_dir=OUTPUT_DIR,
        device="auto",
        dtype="float16",
        method=METHOD,
        on_stage=on_stage,
        on_log=on_log,
        # quantization="4bit",
        # norm_preserve=False,
        # # OR 
        # quantization="8bit",
        # norm_preserve=True,
    )
    if N_DIRECTIONS > 0:
        kwargs["n_directions"] = N_DIRECTIONS
    if REGULARIZATION > 0:
        kwargs["regularization"] = REGULARIZATION
    if REFINEMENT_PASSES > 0:
        kwargs["refinement_passes"] = REFINEMENT_PASSES

    pipeline = AbliterationPipeline(**kwargs)
    result = pipeline.run()

    # result is a PosixPath pointing to the saved model dir
    model_dir = Path(result) if not isinstance(result, Path) else result

    print(f"\n{'='*60}")
    print(f"ABLITERATION COMPLETE")
    print(f"Output: {model_dir}")
    print(f"{'='*60}")

    print(f"\nContents of {model_dir}:")
    for f in sorted(model_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1024**2
            print(f"  {f.relative_to(model_dir)}  ({size_mb:.1f} MB)")

    return model_dir


def generate(model, tokenizer, prompt, max_new_tokens=100):
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


def test(model_dir):
    print(f"\nLoading abliterated model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="auto",
    )

    test_prompts = [
        "The capital of France is",
        "Explain how a neural network works:",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        print(generate(model, tokenizer, prompt))


def main():
    model_dir = abliterate()
    test(model_dir)


if __name__ == "__main__":
    main()
"""
Convert abliterated HuggingFace model to GGUF for LM Studio / llama.cpp.

Requirements:
    git, cmake, python3, pip

What this does:
    1. Clones llama.cpp if not already present
    2. Builds the quantization binary
    3. Converts the HF model to fp16 GGUF
    4. Quantizes to Q4_K_M (recommended) and optionally others

Usage:
    python convert_to_gguf.py
    python convert_to_gguf.py --model_dir path/to/model --quant Q5_K_M
    python convert_to_gguf.py --all_quants   # produces Q4_K_M, Q5_K_M, Q8_0
"""

import argparse
import subprocess
import sys
from pathlib import Path

# --- Config ---
MODEL_DIR = "abliterated"
OUTPUT_DIR = "gguf"
LLAMA_CPP_DIR = "llama.cpp"

# Quantization options (in order of size vs quality tradeoff):
#   Q4_K_M  — best balance, recommended for LM Studio (~2.0 GB for 3B)
#   Q5_K_M  — slightly better quality, larger (~2.4 GB)
#   Q8_0    — near-lossless, large (~3.2 GB)
#   Q2_K    — very small, noticeable quality loss
DEFAULT_QUANT = "Q4_K_M"
ALL_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def run(cmd: list[str], cwd: str | None = None):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result


def clone_llama_cpp():
    if Path(LLAMA_CPP_DIR).exists():
        print(f"llama.cpp already exists at {LLAMA_CPP_DIR}, skipping clone.")
        return
    run(["git", "clone", "https://github.com/ggerganov/llama.cpp", LLAMA_CPP_DIR, "--depth=1"])


def build_llama_cpp():
    build_dir = Path(LLAMA_CPP_DIR) / "build"
    quantize_bin = build_dir / "bin" / "llama-quantize"

    if quantize_bin.exists():
        print(f"llama-quantize already built at {quantize_bin}, skipping build.")
        return str(quantize_bin)

    build_dir.mkdir(exist_ok=True)
    run(["cmake", "..", "-DLLAMA_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"], cwd=str(build_dir))
    run(["cmake", "--build", ".", "--config", "Release", "-j", "4"], cwd=str(build_dir))

    if not quantize_bin.exists():
        raise FileNotFoundError(f"Build failed: {quantize_bin} not found")

    return str(quantize_bin)


def install_conversion_deps():
    run(["uv", "pip", "install", "gguf", "sentencepiece", "tiktoken", "transformers"])


def convert_to_fp16_gguf(model_dir: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fp16_path = out / "model-fp16.gguf"
    if fp16_path.exists():
        print(f"fp16 GGUF already exists at {fp16_path}, skipping conversion.")
        return fp16_path

    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"{convert_script} not found. Make sure llama.cpp cloned correctly."
        )

    run([
        sys.executable, str(convert_script),
        model_dir,
        "--outtype", "f16",
        "--outfile", str(fp16_path),
    ])

    print(f"\nfp16 GGUF saved: {fp16_path} ({fp16_path.stat().st_size / 1024**3:.2f} GB)")
    return fp16_path


def quantize(quantize_bin: str, fp16_path: Path, quant_type: str, output_dir: str) -> Path:
    out = Path(output_dir)
    model_name = Path(fp16_path).stem.replace("-fp16", "")
    quant_path = out / f"{model_name}-{quant_type.lower()}.gguf"

    if quant_path.exists():
        print(f"{quant_type} already exists at {quant_path}, skipping.")
        return quant_path

    run([quantize_bin, str(fp16_path), str(quant_path), quant_type])
    print(f"\n{quant_type} GGUF saved: {quant_path} ({quant_path.stat().st_size / 1024**3:.2f} GB)")
    return quant_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--quant", default=DEFAULT_QUANT, help="e.g. Q4_K_M, Q5_K_M, Q8_0")
    parser.add_argument("--all_quants", action="store_true", help="Build all quant sizes")
    parser.add_argument("--skip_build", action="store_true", help="Skip cmake build (if already built)")
    args = parser.parse_args()

    print(f"Model dir:  {args.model_dir}")
    print(f"Output dir: {args.output_dir}")

    clone_llama_cpp()
    install_conversion_deps()

    quantize_bin = None
    if not args.skip_build:
        quantize_bin = build_llama_cpp()
    else:
        quantize_bin = str(Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize")

    fp16_path = convert_to_fp16_gguf(args.model_dir, args.output_dir)

    quants = ALL_QUANTS if args.all_quants else [args.quant]
    produced = []
    for q in quants:
        p = quantize(quantize_bin, fp16_path, q, args.output_dir)
        produced.append(p)

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    for p in produced:
        print(f"  {p}  ({p.stat().st_size / 1024**3:.2f} GB)")
    print(f"\nTo use in LM Studio:")
    print(f"  Copy the .gguf file(s) to your LM Studio models folder.")
    print(f"  Default: ~/.lmstudio/models/local/")


if __name__ == "__main__":
    main()
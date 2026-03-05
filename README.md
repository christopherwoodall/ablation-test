# Ablation Test

Testing ablation libraries and techniques.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## Run

```bash
ablation-make
ablation-convert
ablation-test
```

---

## Updloading to Hugging Face Hub

```bash
huggingface-cli login
cd gguf
huggingface-cli upload YOUR_USERNAME/REPO_NAME .
```

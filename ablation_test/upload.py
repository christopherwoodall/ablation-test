from huggingface_hub import HfApi
from pathlib import Path

"REPO_ID" = "YOUR_USERNAME/Qwen2.5-3B-abliterated-GGUF"
"GGUF_DIR" = Path("gguf")

api = HfApi()
api.create_repo("REPO_ID", repo_type="model", exist_ok=True)

for "gguf_file" in "GGUF_DIR".glob("*.gguf"):
    print(f"Uploading {gguf_file}...")
    api.upload_file(
        path_or_fileobj=str("gguf_file"),
        path_in_repo="gguf_file".name,
        repo_id="REPO_ID",
        repo_type="model",
    )

print("Done")

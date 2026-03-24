# first do pip install --no-cache-dir huggingface_hub==0.23.5

from huggingface_hub import snapshot_download
HF_TOKEN = "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"
repo     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dst      = "/basemodel/model"

snapshot_download(
    repo_id=repo,
    local_dir=dst,
    token=HF_TOKEN,
    allow_patterns=["*.safetensors","*.json","*.py","tokenizer*","config*"]
)
print("Downloaded to", dst)

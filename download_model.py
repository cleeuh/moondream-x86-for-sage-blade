import os

MOONDREAM_MODEL_DATA_DIR = "./huggingface"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HOME"] = MOONDREAM_MODEL_DATA_DIR
os.environ["HF_MODULES_CACHE"] = MOONDREAM_MODEL_DATA_DIR

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vikhyatk/moondream2",
    revision="2025-04-14",
    repo_type="model",
    local_dir="./moondream2_complete",
    # trust_remote_code=True,
    force_download=True,
    # local_dir_use_symlinks=False,  # Avoid symlinks for true offline use
)
import os

MOONDREAM_MODEL_DATA_DIR = "./huggingface"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HOME"] = MOONDREAM_MODEL_DATA_DIR
os.environ["HF_MODULES_CACHE"] = MOONDREAM_MODEL_DATA_DIR

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vikhyatk/moondream2",
    revision="2025-04-14",
    repo_type="model",
)

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-04-14",
    trust_remote_code=True,
    local_files_only=True,
)
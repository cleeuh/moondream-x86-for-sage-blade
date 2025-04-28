import os

MOONDREAM_MODEL_DATA_DIR = "../huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = MOONDREAM_MODEL_DATA_DIR
os.environ["HF_MODULES_CACHE"] = MOONDREAM_MODEL_DATA_DIR

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image



def test_gpu():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

        # Try to allocate a tensor on the GPU
        x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
        print("Tensor on GPU:", x)
    else:
        print("No GPU available. Running on CPU.")

test_gpu()




model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-04-14",

    trust_remote_code=True,
    local_files_only=True,
    device_map={"": "cuda"}
)

image = Image.open(r"./sample4.jpg")

# Captioning
print("Short caption:")
print(model.caption(image, length="short")["caption"])

print("Normal caption:")
print(model.caption(image, length="normal")["caption"])

print("Long caption:")
print(model.caption(image, length="long")["caption"])

# for t in model.caption(image, length="normal", stream=True)["caption"]:
#     # Streaming generation example, supported for caption() and detect()
#     print(t, end="", flush=True)


# Visual Querying
print("\nVisual query:")
print(model.query(image, "Is there a natural disaster and are you sure?")["answer"])
print(model.query(image, "Is there an eruption?")["answer"])
# Object Detection
print("\nObject detection: 'face'")
objects = model.detect(image, "face")["objects"]
print(f"Found {len(objects)} face(s)")

# Pointing
print("\nPointing: 'smoke'")
points = model.point(image, "smoke")["points"]
print(f"Found {len(points)} smoke(s)")




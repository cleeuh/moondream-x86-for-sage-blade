import torch

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

if __name__ == "__main__":
    test_gpu()
import torch

if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use the GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. PyTorch will use the CPU.")
    device = torch.device("cpu")

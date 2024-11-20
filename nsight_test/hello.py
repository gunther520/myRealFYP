import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create large tensors on the GPU
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# Perform matrix multiplication on the GPU
c = torch.matmul(a, b)

print(c)
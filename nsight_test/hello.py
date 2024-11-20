import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create tensors on the GPU
a = torch.tensor([1, 2, 3], device=device)
b = torch.tensor([4, 5, 6], device=device)

# Perform operations on the GPU
c = a + b

print(c)
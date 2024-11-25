import torch
import torch.cuda.profiler as profiler

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)
# Perform matrix multiplication on the GPU
with torch.autograd.profiler.emit_nvtx():
    profiler.start()
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.randn(1000, 1000, device=device)
    d = a @ b
    e = a @ b
    profiler.stop()

print("Done")
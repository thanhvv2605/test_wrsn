import torch
import time
print("MPS available:", torch.backends.mps.is_available())

torch.backends.cudnn.benchmark = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.rand((10000, 10000), device=device)
start = time.time()
y = torch.matmul(x, x)
print("Operation completed in: ", time.time() - start, "seconds")
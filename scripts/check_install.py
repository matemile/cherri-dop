from importlib.metadata import version

import torch
import anemoi.training

print(f"anemoi-training: {version('anemoi-training')}")
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

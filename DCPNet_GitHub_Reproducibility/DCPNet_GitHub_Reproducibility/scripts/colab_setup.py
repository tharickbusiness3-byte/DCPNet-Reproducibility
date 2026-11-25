from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
print("Drive mounted.")
print("GPU verification:")
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

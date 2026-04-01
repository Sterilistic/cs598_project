import torch

print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0))
print("Supported arch list:", torch.cuda.get_arch_list())

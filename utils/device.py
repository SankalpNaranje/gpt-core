# core/device.py

import torch

def get_device(print_device=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if print_device:
        print(f"Using {device} device.")

    return device
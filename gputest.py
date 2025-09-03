import os
import torch

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def list_available_gpus():
    if not torch.cuda.is_available():
        print("No GPU available")
        return []

    num_gpus = torch.cuda.device_count()
    gpus = []
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        gpus.append((i, name))
    return gpus

if __name__ == "__main__":
    gpus = list_available_gpus()
    if gpus:
        for idx, name in gpus:
            print(f"GPU {idx}: {name}")
    else:
        print("No GPUs found.")
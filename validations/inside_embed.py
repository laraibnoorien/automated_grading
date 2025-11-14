import torch

data = torch.load("embeddings_library.pt")

def print_keys(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            print(prefix + k)
            print_keys(v, prefix + "  ")

print_keys(data)

import torch
from parameter_set import *

def mask_tril(data):
    mask = 1 - torch.tril(torch.ones(1, mod, mod, dtype=torch.long))
    mask = mask > 0
    mask = (mask == 1).unsqueeze(dim=1)
    return mask

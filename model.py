import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AttReg(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        pass
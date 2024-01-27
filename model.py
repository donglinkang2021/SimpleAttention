import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        raise NotImplementedError

class Linear_Reg_Plane(BaseModel):
    """Linear regression model with one hidden layer"""
    def __init__(self, input_dim, n_hid, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, output_dim)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class Heads_Reg_Plane(BaseModel):
    """Regression model with multi-head between-heads attention"""
    def __init__(self, input_dim, n_embd, n_head, output_dim):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.embed = nn.Linear(input_dim, n_embd)
        self.pred = nn.Linear(n_embd, output_dim)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")
    
    def get_wei(self, x):
        x = self.embed(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        wei = x @ x.transpose(-2, -1) / math.sqrt(self.n_embd)
        wei = F.softmax(wei, dim=-1)
        return wei

    def forward(self, x):
        x = self.embed(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        attention = F.scaled_dot_product_attention(x, x, x)
        attention = rearrange(attention, 'B nh hs -> B (nh hs)')
        return self.pred(attention)
    

class Batchs_Reg_Plane(BaseModel):
    """Regression model with between-batchs attention"""
    def __init__(self, input_dim, n_embd, n_batchs, output_dim):
        super().__init__()
        self.n_embd = n_embd
        self.n_batchs = n_batchs
        self.embed = nn.Linear(input_dim, n_embd)
        self.pred = nn.Linear(n_embd, output_dim)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def get_wei(self, x):
        assert x.shape[0] % self.n_batchs == 0, "batch size must be divisible by n_batchs"
        x = self.embed(x)
        x = rearrange(x, '(nB Bs) d -> Bs nB d', nB=self.n_batchs)
        wei = x @ x.transpose(-2, -1) / math.sqrt(self.n_embd)
        wei = F.softmax(wei, dim=-1)
        return wei

    def forward(self, x):
        x = self.embed(x)
        x = rearrange(x, '(nB Bs) d -> Bs nB d', nB=self.n_batchs)
        attention = F.scaled_dot_product_attention(x, x, x)
        attention = rearrange(attention, 'Bs nB d -> (nB Bs) d')
        return self.pred(attention)
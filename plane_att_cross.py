import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from dataset import regress_plane
from einops import rearrange

np.random.seed(2024)
torch.manual_seed(2024)

key_samples = 250
query_samples = 250
test_samples = 500
noise = 0.2

batch_size = 64
num_epochs = 3
eval_interval = 1
learning_rate = 1e-2

input_dim = 2 
n_embd = 8
n_head = 4

class PlaneDataset(Dataset):
    def __init__(self, num_samples, noise):
        super().__init__()
        self.num_samples = num_samples
        self.noise = noise
        self.x, self.y, self.label = regress_plane(num_samples, noise)
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        self.X = torch.stack([self.x, self.y], dim=1)
        self.label = torch.from_numpy(self.label).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.label[idx]

query_set = PlaneDataset(query_samples, noise) # we use other dataset for query later
key_set = PlaneDataset(key_samples, noise)
value_set = PlaneDataset(test_samples, noise)
queryloader = DataLoader(query_set, batch_size=batch_size, shuffle=True)
keyloader = DataLoader(key_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(value_set, batch_size=batch_size, shuffle=True)


class PlaneModel(nn.Module):
    def __init__(self, input_dim, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.decoder = nn.Linear(input_dim, n_embd)
        self.encoder = nn.Linear(input_dim, n_embd)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, query, key, value):
        # decoder
        # 1. decoder embedding
        q = self.decoder(query) 
        # 2. self attention (neighbourhood aggregation)
        q = rearrange(q, 'B (nh hs) -> B nh hs', nh=self.n_head)
        q = F.scaled_dot_product_attention(q, q, q)
        q = rearrange(q, 'B nh hs -> B (nh hs)')
        # encoder
        # 1. encoder embedding
        k = self.encoder(key)
        # 2. self attention (neighbourhood aggregation)
        k = rearrange(k, 'B (nh hs) -> B nh hs', nh=self.n_head)
        k = F.scaled_dot_product_attention(k, k, k)
        k = rearrange(k, 'B nh hs -> B (nh hs)')
        
        v = value.unsqueeze(1)  # value
        # v = self.model(k) we can use other well-trained model to get the value
        # here we just provide ground truth
        # cross attention
        wei = q @ k.transpose(-1, -2) / math.sqrt(self.n_embd)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out      
    
model = PlaneModel(input_dim, n_embd, n_head)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for name, loader in [('train', queryloader), ('val', testloader)]:
        losses = []
        for x, y in loader:
            for key_x, key_y in keyloader:
                y_pred = model(x, key_x, key_y)
                loss = criterion(y_pred, y.unsqueeze(1))
                losses.append(loss.item())
        out[name] = np.mean(losses)
    model.train()
    return out


n_query_batches = len(queryloader)
n_key_batches = len(keyloader)
for epoch in range(num_epochs):
    for i, (query_x, query_y) in enumerate(queryloader):
        iter_i = epoch * n_query_batches + i
        for j, (key_x, key_y) in enumerate(keyloader):
            iter_j = iter_i * n_key_batches + j

            if iter_j % eval_interval == 0 or iter_j == num_epochs * n_query_batches * n_key_batches - 1:
                losses = estimate_loss()
                print(f"step {iter_j}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
            y_pred = model(query_x, key_x, key_y)
            loss = criterion(y_pred, query_y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

"""
number of parameters: 0.000048 M 
step 0: train loss 0.1749, val loss 0.1580
step 1: train loss 0.1724, val loss 0.1566
step 2: train loss 0.1710, val loss 0.1542
step 3: train loss 0.1632, val loss 0.1473
step 4: train loss 0.1530, val loss 0.1374
step 5: train loss 0.1420, val loss 0.1278
step 6: train loss 0.1246, val loss 0.1123
step 7: train loss 0.1056, val loss 0.0970
step 8: train loss 0.0851, val loss 0.0777
step 9: train loss 0.0679, val loss 0.0594
step 10: train loss 0.0478, val loss 0.0457
step 11: train loss 0.0324, val loss 0.0310
step 12: train loss 0.0221, val loss 0.0215
step 13: train loss 0.0173, val loss 0.0174
step 14: train loss 0.0123, val loss 0.0149
step 15: train loss 0.0126, val loss 0.0164
step 16: train loss 0.0140, val loss 0.0181
step 17: train loss 0.0167, val loss 0.0208
step 18: train loss 0.0195, val loss 0.0241
step 19: train loss 0.0217, val loss 0.0263
step 20: train loss 0.0241, val loss 0.0286
step 21: train loss 0.0258, val loss 0.0290
step 22: train loss 0.0256, val loss 0.0305
step 23: train loss 0.0271, val loss 0.0291
step 24: train loss 0.0244, val loss 0.0277
step 25: train loss 0.0238, val loss 0.0261
step 26: train loss 0.0222, val loss 0.0250
step 27: train loss 0.0201, val loss 0.0222
step 28: train loss 0.0187, val loss 0.0205
step 29: train loss 0.0163, val loss 0.0189
step 30: train loss 0.0146, val loss 0.0186
step 31: train loss 0.0132, val loss 0.0169
step 32: train loss 0.0132, val loss 0.0148
step 33: train loss 0.0108, val loss 0.0137
step 34: train loss 0.0134, val loss 0.0134
step 35: train loss 0.0126, val loss 0.0141
step 36: train loss 0.0110, val loss 0.0128
step 37: train loss 0.0112, val loss 0.0137
step 38: train loss 0.0111, val loss 0.0149
step 39: train loss 0.0120, val loss 0.0132
step 40: train loss 0.0130, val loss 0.0145
step 41: train loss 0.0118, val loss 0.0136
step 42: train loss 0.0135, val loss 0.0147
step 43: train loss 0.0127, val loss 0.0152
step 44: train loss 0.0125, val loss 0.0142
step 45: train loss 0.0122, val loss 0.0128
step 46: train loss 0.0128, val loss 0.0133
step 47: train loss 0.0123, val loss 0.0128
"""
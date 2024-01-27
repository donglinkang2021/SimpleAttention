import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
n_embd = 32
output_dim = 1

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
    def __init__(self, input_dim, n_embd):
        super().__init__()
        self.embed = nn.Linear(input_dim, n_embd)
        self.pred = nn.Linear(n_embd, output_dim)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, query, key):
        q = self.embed(query)
        k = self.embed(key)
        v = self.pred(k)
        wei = q @ k.transpose(-2, -1)
        wei = F.softmax(wei, dim=-1)
        return wei @ v        
    
model = PlaneModel(input_dim, n_embd)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for name, loader in [('train', queryloader), ('val', testloader)]:
        losses = []
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y.unsqueeze(1))
            losses.append(loss.item())
        out[name] = np.mean(losses)
    model.train()
    return out


n_batches = len(queryloader)
for epoch in range(num_epochs):
    for i, (query_x, query_y) in enumerate(queryloader):
        iter = epoch * n_batches + i

        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        y_pred = model(x)
        loss = criterion(y_pred, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
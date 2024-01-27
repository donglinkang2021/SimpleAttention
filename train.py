import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import regress_gaussian
from model import Heads_Reg_Gaussian

np.random.seed(2024)
torch.manual_seed(2024)

train_samples = 500
val_samples = 500
noise = 0.01 # a good para to test the model performance

batch_size = 64
num_epochs = 30
eval_interval = 10
learning_rate = 3e-2

input_dim = 2 
n_embd = 64 # n_embd:param 8:33 16:65 32:129 64:257
n_head = 64
output_dim = 1


class RegressionDataset(Dataset):
    def __init__(self, num_samples, noise):
        super().__init__()
        self.num_samples = num_samples
        self.noise = noise
        self.x, self.y, self.label = regress_gaussian(num_samples, noise)
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        self.X = torch.stack([self.x, self.y], dim=1)
        self.label = torch.from_numpy(self.label).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.label[idx]

trainset = RegressionDataset(train_samples, noise)
valset = RegressionDataset(val_samples, noise)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    

model = Heads_Reg_Gaussian(input_dim, n_embd, n_head, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for name, loader in [('train', trainloader), ('val', valloader)]:
        losses = []
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y.unsqueeze(1))
            losses.append(loss.item())
        out[name] = np.mean(losses)
    model.train()
    return out

iter_list = []
train_losses = []
val_losses = []
n_batches = len(trainloader)
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(trainloader):
        iter = epoch * n_batches + i

        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            iter_list.append(iter)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

        y_pred = model(x)
        loss = criterion(y_pred, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
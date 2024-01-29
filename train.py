import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import classify_spiral_data, regress_gaussian
from model import Heads_Reg
from model import Linear_Reg_Gaussian

np.random.seed(2024)
torch.manual_seed(2024)

task = "regression" # "regression" or "classification"

train_samples = 500
val_samples = 500
noise = 0.01 # a good para to test the model performance

batch_size = 64
num_epochs = 30
eval_interval = 10
learning_rate = 3e-2

input_dim = 2 
n_embd = 8 # reg att: n_embd:param 4:17 8:33 16:65 32:129 64:257; cls att: n_embd:param 4:22 8:42 16:82 32:162 64:322
n_head = 4

if task == "regression":
    output_dim = 1
else:
    output_dim = 2

is_train_heads = False 
# True: train heads; False: train linear regression

class RegressionDataset(Dataset):
    def __init__(self, num_samples, noise):
        super().__init__()
        self.num_samples = num_samples
        self.noise = noise
        self.x, self.y, self.label = regress_gaussian(num_samples, noise)
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        self.X = torch.stack([self.x, self.y], dim=1)
        self.label = torch.from_numpy(self.label).float().unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.label[idx]

class ClassificationDataset(Dataset):
    def __init__(self, num_samples, noise):
        super().__init__()
        self.num_samples = num_samples
        self.noise = noise
        self.x, self.y, self.label = classify_spiral_data(num_samples, noise)
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        self.X = torch.stack([self.x, self.y], dim=1)
        self.label = torch.from_numpy((self.label + 1) // 2).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.label[idx]

if task == "regression":
    trainset = RegressionDataset(train_samples, noise)
    valset = RegressionDataset(val_samples, noise)
else:
    trainset = ClassificationDataset(train_samples, noise)
    valset = ClassificationDataset(val_samples, noise)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    
if is_train_heads:
    model = Heads_Reg(input_dim, n_embd, n_head, output_dim)
else:
    model = Linear_Reg_Gaussian(input_dim, n_embd, output_dim)

if task == "regression":
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for name, loader in [('train', trainloader), ('val', valloader)]:
        losses = []
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
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
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
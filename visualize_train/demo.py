import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import regress_gaussian
from model import Linear_Reg_Gaussian
from einops import rearrange
import matplotlib.pyplot as plt

np.random.seed(2024)
torch.manual_seed(2024)

train_samples = 500
val_samples = 500
noise = 0.01 # a good para to test the model performance

batch_size = 64
num_epochs = 25
eval_interval = 2
learning_rate = 3e-2

input_dim = 2 
n_embd = 8 # att: n_embd:param 4:17 8:33 16:65 32:129 64:257
# n_head = 4
output_dim = 1

# model_name = f"heads{n_head}_embed{n_embd}_no_act"
model_name = f"linear_hid2_embed{n_embd}_tanh"
dataset_name = "regress_gaussian"

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
    

model = Linear_Reg_Gaussian(input_dim, n_embd, output_dim)
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

@torch.no_grad()
def draw_perbatch(iter, dataset, model_name, dataset_name):
    h = 0.02
    x_min, x_max = dataset.x.min() - .05, dataset.x.max() + .05
    y_min, y_max = dataset.y.min() - .05, dataset.y.max() + .05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    X = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    batch_size = 64
    embed = torch.zeros(X.shape[0], n_embd)
    embed_att = torch.zeros(X.shape[0], n_embd)
    Z = torch.zeros(X.shape[0], 1)
    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, X.shape[0])
        batch_x = X[i:end]
        # embed_tmp = model.embed(batch_x)
        # embed[i:end] = embed_tmp
        # embed_tmp = rearrange(embed_tmp, 'B (nh hs) -> B nh hs', nh=n_head)
        # attention = F.scaled_dot_product_attention(embed_tmp, embed_tmp, embed_tmp)
        # attention = rearrange(attention, 'B nh hs -> B (nh hs)')
        # embed_att[i:end] = attention
        batch_x = F.tanh(model.fc1(batch_x))
        embed[i:end] = batch_x
        batch_x = F.tanh(model.fc2(batch_x))
        embed_att[i:end] = batch_x
        Z[i:end] = model.fc3(batch_x)

    Z = Z.reshape(xx.shape)
    embed = rearrange(embed, '(h w) d -> d h w', h=xx.shape[0])
    embed_att = rearrange(embed_att, '(h w) d -> d h w', h=xx.shape[0])
    fig, axs = plt.subplots(2, n_embd, figsize=(4 * n_embd, 8))
    for i in range(n_embd):
        axs[0, i].contourf(xx, yy, embed[i].detach().numpy(), cmap=plt.cm.bwr, alpha=0.5)
        # axs[0, i].scatter(dataset.x, dataset.y, c=dataset.label, cmap=plt.cm.bwr)
        # axs[0, i].set_title(f"Embedding {i}")
        axs[0, i].set_title(f"1th layer Hidden Unit {i}")
        axs[1, i].contourf(xx, yy, embed_att[i].detach().numpy(), cmap=plt.cm.bwr, alpha=0.5)
        # axs[1, i].scatter(dataset.x, dataset.y, c=dataset.label, cmap=plt.cm.bwr)
        # axs[1, i].set_title(f"Attention {i}")
        axs[1, i].set_title(f"2th layer Hidden Unit {i}")
    # plt.suptitle(f"Embedding and Attention for model:{model_name}, dataset:{dataset_name}, steps:{iter}")
    plt.suptitle(f"Hidden Unit for model:{model_name}, dataset:{dataset_name}, steps:{iter}")
    plt.savefig(f"hid/{model_name}_{dataset_name}_{iter}_embed_att.png")
    plt.close(fig)
    # plt.show()
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.detach().numpy(), cmap=plt.cm.bwr, alpha=0.5)
    plt.scatter(dataset.x, dataset.y, c=dataset.label, cmap=plt.cm.bwr)
    plt.title(f"Decision boundary for model:{model_name}, dataset:{dataset_name}, steps:{iter}")
    plt.colorbar()
    plt.savefig(f"fin/{model_name}_{dataset_name}_{iter}_decision_boundary.png")
    plt.close()
    

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
            draw_perbatch(
                iter, 
                valset, 
                model_name, 
                dataset_name
            )

        y_pred = model(x)
        loss = criterion(y_pred, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
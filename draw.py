import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def show_attention(
        attention,
        xlabel = 'Keys',
        ylabel = 'Queries',
        title = 'Attention weights',
        figsize=(5, 5),
        cmap = 'Reds'
    ):
    """
    visualize attention weights
    
    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (m, n)
    """

    fig = plt.figure(figsize = figsize)

    pcm = plt.imshow(
        attention, 
        cmap = cmap
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.colorbar(pcm, shrink=0.7)
    plt.show()


def show_attention_batch(
        attention,
        xlabel='Keys',
        ylabel='Queries',
        title='Attention weights',
        figsize=(5, 5),
        cmap='Reds'
    ):
    """
    visualize attention weights in batch

    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (b, m, n)
    """

    # Get the batch size
    batch_size, _, _ = attention.shape

    ncols = int(np.sqrt(batch_size))
    nrows = int(np.ceil(batch_size / ncols))

    # Create subplots for each attention matrix in the batch
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    # Iterate through each attention matrix in the batch
    for i in range(nrows):
        for j in range(ncols):
            # Get the attention matrix
            ax = axs[i, j]
            ax.imshow(
                attention[i * ncols + j],
                cmap=cmap
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} - Batch {i * ncols + j + 1}')

    plt.tight_layout()
    plt.show()

def draw_dataset(dataset:Dataset, title:str):
    """
    draw dataset
    """
    assert dataset.x != None and dataset.y != None and dataset.label != None, "dataset must have x, y, label"
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset.x, dataset.y, c=dataset.label, cmap=plt.cm.bwr)
    if title != None:
        plt.title(title)
    plt.colorbar()
    plt.show()

@torch.no_grad()
def draw_decision_boundary(model:nn.Module, dataset:Dataset, title:str, is_save:bool=False):
    """
    draw decision boundary of a model on a 2D dataset
    """
    assert dataset.x != None and dataset.y != None and dataset.label != None, "dataset must have x, y, label"
    h = 0.02
    x_min, x_max = dataset.x.min() - .05, dataset.x.max() + .05
    y_min, y_max = dataset.y.min() - .05, dataset.y.max() + .05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    X = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    # to run this my memory is not enough
    # Z = model(X)
    
    batch_size = 64
    Z = torch.zeros(X.shape[0], 1)
    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, X.shape[0])
        batch_x = X[i:end]
        Z[i:end] = model(batch_x)
    
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.detach().numpy(), cmap=plt.cm.bwr, alpha=0.5)
    plt.scatter(dataset.x, dataset.y, c=dataset.label, cmap=plt.cm.bwr)
    if title != None:
        plt.title(title)
    plt.colorbar()
    if is_save:
        png_name = title.replace(" ", "_").replace(":", "_").replace(",", "")
        plt.savefig(f"{png_name}.png")
    plt.show()

def draw_loss(iter_list, train_loss, val_loss, title:str, is_save:bool=False):
    """
    draw loss
    """
    plt.figure(figsize=(8, 6))
    plt.plot(iter_list, train_loss, label='train')
    plt.plot(iter_list, val_loss, label='val')
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('loss')
    if title != None:
        plt.title(title)
    if is_save:
        png_name = title.replace(" ", "_").replace(":", "_").replace(",", "")
        plt.savefig(f"{png_name}.png")
    plt.show()
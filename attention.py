import numpy as np
from typing import Tuple

def norm1d(
        data:np.ndarray, 
        axis:np.ndarray
    ) -> np.ndarray:
    """Normalize data to have zero mean and unit standard deviation along the specified axis."""
    return (data - data.mean(axis=axis, keepdims=True)) / data.std(axis=axis, keepdims=True)

def scale_dot_product(
        query:np.ndarray, 
        key:np.ndarray, 
        value:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled Dot Product Attention

    Args:
        query (np.ndarray): Query matrix  (l1, d)
        key (np.ndarray): Key matrix      (l2, d)
        value (np.ndarray): Value matrix  (l2, d)

    Returns:
        np.ndarray: Attention matrix      (l1, d)
        np.ndarray: Weight matrix         (l1, l2)
    """
    # Get the dimension of the query matrix
    dim = query.shape[-1]

    # Compute the dot product
    dot_product = np.matmul(query, key.T)

    # Scale the dot product
    wei = dot_product / np.sqrt(dim)

    # Compute the softmax
    attention_wei = np.exp(wei) / np.sum(np.exp(wei), axis=-1, keepdims=True)

    # Compute the attention
    attention = np.matmul(attention_wei, value)

    return attention, attention_wei

def querynorm_dot_prduct(
        query:np.ndarray, 
        key:np.ndarray, 
        value:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Query Normalized Dot Product Attention

    Args:
        query (np.ndarray): Query matrix  (l1, d)
        key (np.ndarray): Key matrix      (l2, d)
        value (np.ndarray): Value matrix  (l2, d)

    Returns:
        np.ndarray: Attention matrix      (l1, d)
        np.ndarray: Weight matrix         (l1, l2)
    """

    # Add LayerNorm to the query dim
    if query.shape[-1] > 1:
        # Normalize the query
        query = norm1d(query, axis=-1)

    # Compute the dot product
    wei = np.matmul(query, key.T)

    # Compute the softmax
    attention_wei = np.exp(wei) / np.sum(np.exp(wei), axis=-1, keepdims=True)

    # Compute the attention
    attention = np.matmul(attention_wei, value)

    return attention, attention_wei

def kernel_func(dists, kernel="Gaussian"):
    """Kernel function
    
    Usage
    -----

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(
    >>>     1, 4, 
    >>>     sharey=True, 
    >>>     figsize=(12, 3)
    >>> )
    >>> names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
    >>> x = np.arange(-2.5, 2.5, 0.1)
    >>> for name, ax in zip(names, axes):
    >>>     ax.plot(x, kernel_func(x, name))
    >>>     ax.set_xlabel(name)
    >>> plt.show()
    """
    # compute the kernel function
    if kernel == "Gaussian":
        wei = np.exp(-dists**2)
    elif kernel == "Boxcar":
        wei = np.abs(dists) < 1.0
    elif kernel == "Constant":
        wei = np.ones_like(dists)
    elif kernel == "Epanechikov":
        wei = np.maximum(1 - np.abs(dists), np.zeros_like(dists)) # np.maximum is element-wise max
    else:
        raise ValueError("Invalid kernel function")
    return wei

def nadaraya_watson(
        query:np.ndarray, 
        key:np.ndarray, 
        value:np.ndarray,
        kernel = "Gaussian"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Nadaraya-Watson Kernel Regression

    Args:
        query (np.ndarray): Query matrix  (l1, d)
        key (np.ndarray): Key matrix      (l2, d)
        value (np.ndarray): Value matrix  (l2, d)
        kernel (str, optional): Kernel function. Defaults to "Gaussian".
        - names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')

    Returns:
        np.ndarray: Attention matrix      (l1, d)
        np.ndarray: Weight matrix         (l1, l2)
    """
    
    # compute the pairwise distances between every query and key

    if query.shape[-1] == 1:
        dists = query - key.T
    else:
        dists = np.sqrt(np.sum((query[:, None, :] - key[None, :, :])**2, axis=-1))
    
    wei = kernel_func(dists, kernel)
    
    # compute the attention weights
    attention_wei = wei / wei.sum(0) 
    
    # compute the predicted value
    attention = np.matmul(attention_wei, value)
    return attention, attention_wei
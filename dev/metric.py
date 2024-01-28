import numpy as np

def mse(
        y_true:np.ndarray,
        y_pred:np.ndarray
    ) -> np.ndarray:
    """Mean Squared Error

    Args:
        y_true (np.ndarray): Ground truth
        y_pred (np.ndarray): Predicted values

    Returns:
        np.ndarray: Mean squared error
    """
    return np.mean((y_true - y_pred)**2)

def mae(
        y_true:np.ndarray,
        y_pred:np.ndarray
    ) -> np.ndarray:
    """Mean Absolute Error

    Args:
        y_true (np.ndarray): Ground truth
        y_pred (np.ndarray): Predicted values

    Returns:
        np.ndarray: Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))

def cross_entropy(
        labels:np.ndarray,
        preds:np.ndarray
    ) -> np.ndarray:
    """Cross Entropy

    Args:
        labels (np.ndarray): Ground truth class index      (n,)
        preds (np.ndarray): Predicted distribution matrix  (n, C)

    Returns:
        np.ndarray: Cross entropy     
    """
    return -np.mean(np.log(preds[range(len(labels)), labels.tolist()]))
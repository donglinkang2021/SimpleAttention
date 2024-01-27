import numpy as np
from typing import Tuple

def regress_plane(
        num_samples:int, 
        noise:float, 
        radius:int = 6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points on a plane with a label
    corresponding to the distance from the origin.
    """
    def get_label(x, y):
        return (x + y) / (2 * radius)

    x = np.random.uniform(-radius, radius, num_samples)
    y = np.random.uniform(-radius, radius, num_samples)
    noise_x = np.random.uniform(-radius, radius, num_samples) * noise
    noise_y = np.random.uniform(-radius, radius, num_samples) * noise
    label = get_label(x + noise_x, y + noise_y)

    return x, y, label

def regress_gaussian(
        num_samples:int, 
        noise:float, 
        radius:int = 6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points with designed gaussian centers
    """
    def label_scale(distance):
        # distance less, label more
        # more probability of being closed to the center
        return distance * -0.1 + 1 

    gaussians = np.array([
        [-4, 2.5, 1],
        [0, 2.5, -1],
        [4, 2.5, 1],
        [-4, -2.5, -1],
        [0, -2.5, 1],
        [4, -2.5, -1]
    ])
    gaussian_xy = gaussians[:, :2] # (num_gaussians, 2)
    gaussian_sign = gaussians[:, 2]

    x = np.random.uniform(-radius, radius, num_samples)
    y = np.random.uniform(-radius, radius, num_samples)
    noise_x = np.random.uniform(-radius, radius, num_samples) * noise
    noise_y = np.random.uniform(-radius, radius, num_samples) * noise
    tmp_xy = np.array([x + noise_x, y + noise_y]).T # (num_samples, 2)
    # we get the distance between our random points and the gaussian centers
    distance = np.sum((tmp_xy[:, np.newaxis] - gaussian_xy) ** 2, axis=-1) # (num_samples, num_gaussians)
    # select the gaussian center idx with the smallest distance
    idx = distance.argmin(axis=-1) # (num_samples,)
    # get the gaussian sign label for each sample
    label = (label_scale(distance) * gaussian_sign)[np.arange(num_samples), idx]

    return x, y, label

def classify_two_gauss_data(
        num_samples:int, 
        noise:float, 
        radius:int = 6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points with two gaussian centers
    """
    variance = radius * noise + 0.5
    n = num_samples // 2

    def gen_gauss(cx, cy, label):
        x = cx + np.sqrt(variance) * np.random.randn(n)
        y = cy + np.sqrt(variance) * np.random.randn(n)
        label = np.ones(n) * label
        return x, y, label

    x_pos, y_pos, label_pos = gen_gauss(2, 2, 1)
    x_neg, y_neg, label_neg = gen_gauss(-2, -2, -1)

    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    label = np.concatenate([label_pos, label_neg])

    return x, y, label

def classify_spiral_data(
        num_samples:int, 
        noise:float, 
        radius:int = 6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points with two spiral data
    """
    n = num_samples // 2

    def gen_spiral(delta_t, label):
        r = np.linspace(0.0, radius, n)
        t = 1.75 * np.linspace(0.0, 2 * np.pi, n) + delta_t
        x = r * np.sin(t) + np.random.uniform(-1, 1, n) * noise
        y = r * np.cos(t) + np.random.uniform(-1, 1, n) * noise
        label = np.ones(n) * label
        return x, y, label

    x_pos, y_pos, label_pos = gen_spiral(0, 1)
    x_neg, y_neg, label_neg = gen_spiral(np.pi, -1)
    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    label = np.concatenate([label_pos, label_neg])

    return x, y, label

def classify_circle_data(
        num_samples:int, 
        noise:float, 
        radius:int = 6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points with two circle data
    """
    n = num_samples // 2

    def gen_circle(min_r, max_r):
        r = np.random.uniform(min_r, max_r, n)
        t = np.random.uniform(0, 2 * np.pi, n)
        x = r * np.sin(t)
        y = r * np.cos(t)
        noise_x = np.random.uniform(-radius, radius, n) * noise
        noise_y = np.random.uniform(-radius, radius, n) * noise
        tmp_xy = np.array([x + noise_x, y + noise_y]).T # (num_samples, 2)
        distance = np.sqrt(np.sum(tmp_xy ** 2, axis=-1))
        label = np.ones(n) * (distance < (radius * 0.5)) * 2 - 1
        return x, y, label
    
    x_pos, y_pos, label_pos = gen_circle(0, radius * 0.5)
    x_neg, y_neg, label_neg = gen_circle(radius * 0.7, radius)
    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    label = np.concatenate([label_pos, label_neg])

    return x, y, label

def classify_xor_data(
        num_samples:int, 
        noise:float, 
        radius:int = 6,
        padding:float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points with two xor data
    """

    x = np.random.uniform(-radius, radius, num_samples)
    y = np.random.uniform(-radius, radius, num_samples)
    x += ((x > 0) * 2 - 1) * padding
    y += ((y > 0) * 2 - 1) * padding
    noise_x = np.random.uniform(-radius, radius, num_samples) * noise
    noise_y = np.random.uniform(-radius, radius, num_samples) * noise
    label = ((x + noise_x) * (y + noise_y) >= 0) * 2 - 1

    return x, y, label
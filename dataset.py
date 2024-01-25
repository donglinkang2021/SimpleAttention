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

def classify_two_gauss_data(num_samples, noise):
    points = []

    variance = np.interp(noise, [0, 0.5], [0.5, 4])

    def gen_gauss(cx, cy, label):
        for _ in range(num_samples // 2):
            x = normal_random(cx, variance)
            y = normal_random(cy, variance)
            points.append({'x': x, 'y': y, 'label': label})

    gen_gauss(2, 2, 1)
    gen_gauss(-2, -2, -1)

    return points

def classify_spiral_data(num_samples, noise):
    points = []
    n = num_samples // 2

    def gen_spiral(delta_t, label):
        for i in range(n):
            r = i / n * 5
            t = 1.75 * i / n * 2 * np.pi + delta_t
            x = r * np.sin(t) + np.random.uniform(-1, 1) * noise
            y = r * np.cos(t) + np.random.uniform(-1, 1) * noise
            points.append({'x': x, 'y': y, 'label': label})

    gen_spiral(0, 1)
    gen_spiral(np.pi, -1)

    return points

def classify_circle_data(num_samples, noise):
    points = []
    radius = 5

    def get_circle_label(p, center):
        return 1 if dist(p, center) < (radius * 0.5) else -1

    for _ in range(num_samples // 2):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label({'x': x + noise_x, 'y': y + noise_y}, {'x': 0, 'y': 0})
        points.append({'x': x, 'y': y, 'label': label})

    for _ in range(num_samples // 2):
        r = np.random.uniform(radius * 0.7, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label({'x': x + noise_x, 'y': y + noise_y}, {'x': 0, 'y': 0})
        points.append({'x': x, 'y': y, 'label': label})

    return points

def classify_xor_data(num_samples, noise):
    def get_xor_label(p):
        return 1 if p['x'] * p['y'] >= 0 else -1

    points = []

    for _ in range(num_samples):
        x = np.random.uniform(-5, 5)
        padding = 0.3
        x += padding if x > 0 else -padding
        y = np.random.uniform(-5, 5)
        y += padding if y > 0 else -padding
        noise_x = np.random.uniform(-5, 5) * noise
        noise_y = np.random.uniform(-5, 5) * noise
        label = get_xor_label({'x': x + noise_x, 'y': y + noise_y})
        points.append({'x': x, 'y': y, 'label': label})

    return points

def normal_random(mean=0, variance=1):
    return mean + np.sqrt(variance) * np.random.randn()

def dist(x1, y1, x2, y2):
    dx, dy = x1 - x2, y1 - y2
    return np.sqrt(dx*dx + dy*dy)

import numpy as np

def norm(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unit_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def perpendicular(a):
    return np.array([-a[1], a[0]])
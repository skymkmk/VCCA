import numpy as np

from .train import train


def cca(train_set: np.ndarray, train_label: list | np.ndarray):
    return train(train_set, train_label)

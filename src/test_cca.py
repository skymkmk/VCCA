import numpy as np

from .validate import validate


def test_cca(model: list, test_set: np.ndarray, test_label: list | np.ndarray):
    cant_identified = 0
    correct_num = 0
    for idx, sample in enumerate(test_set):
        result = validate(model, sample)
        if result == test_label[idx]:
            correct_num += 1
        else:
            cant_identified += 1
    return correct_num, cant_identified

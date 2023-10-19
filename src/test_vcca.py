import numpy as np

from .validate import validate


def test_vcca(models: list, test_set: np.ndarray, test_label: list | np.ndarray):
    cant_identified = 0
    correct_num = 0
    for idx, sample in enumerate(test_set):
        results = []
        for model in models:
            results.append(validate(model, sample))
        result = max(results, key=results.count)
        if result == test_label[idx]:
            correct_num += 1
        else:
            cant_identified += 1
    return correct_num, cant_identified

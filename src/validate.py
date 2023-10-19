import numpy as np


def validate(model: list, sample: np.ndarray):
    result = None
    result_dot = None
    result_idx = None
    for idx, coverage in enumerate(model):
        if result is None:
            if np.dot(sample, coverage['center']) >= coverage['r']:
                result = coverage['label']
                result_dot = np.dot(sample, coverage['center'])
                result_idx = idx
        else:
            if np.dot(sample, coverage['center']) >= coverage['r']:
                if coverage['label'] != result:
                    if coverage['r'] / np.dot(sample, coverage['center']) > model[result_idx]['count'] / result_dot:
                        result = coverage['label']
                        result_dot = np.dot(sample, coverage['center'])
                        result_idx = idx
    return result

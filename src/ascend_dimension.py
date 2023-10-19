import math

import numpy as np


def ascend_dimension(dataset: np.ndarray):
    norm = [np.linalg.norm(i) for i in dataset]
    r = max(norm)
    ascended = np.array([math.sqrt(r ** 2 - i ** 2) for i in norm])
    result = np.column_stack((dataset, ascended))
    return result

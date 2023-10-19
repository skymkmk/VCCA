import numpy as np

from .train import train


def vcca(train_set: np.ndarray, train_label: list | np.ndarray, model_num: int):
    models = []
    # 训练出 15 个模型进行投票
    for i in range(model_num):
        models.append(train(train_set, train_label))
    return models

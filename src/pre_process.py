import numpy as np
import sklearn.impute

from .ascend_dimension import ascend_dimension


def pre_process(dataset: list, neighbors_num: int):
    # 分割数据集与标签
    dataset = np.array(dataset)
    label = dataset[:, -1]
    dataset = dataset[:, 0:-1]
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] == '?':
                dataset[i][j] = np.nan
    dataset = dataset.astype(np.float64)
    knn = sklearn.impute.KNNImputer(n_neighbors=neighbors_num)
    # 缺失项填充
    dataset = knn.fit_transform(dataset)
    # 归一化
    dataset = sklearn.preprocessing.MinMaxScaler().fit_transform(dataset)
    # 升维
    dataset = ascend_dimension(dataset)
    return dataset, label

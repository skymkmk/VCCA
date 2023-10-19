import random

import numpy as np


def _delete_in_range(in_range: np.ndarray, train_set: np.ndarray, train_label: np.ndarray, omega: np.ndarray):
    in_range_index = [int(i) for i in in_range[:, 0].tolist()]
    in_range_samples = np.row_stack((np.array([train_set[i] for i in in_range_index]), omega))
    train_set = np.delete(train_set, in_range_index, axis=0)
    train_label = np.delete(train_label, in_range_index, axis=0)
    return in_range_samples, train_set, train_label


def train(train_set: np.ndarray, train_label: list | np.ndarray):
    model = []
    while len(train_set) != 0:
        # omega 为覆盖中心
        omega_index = random.randint(0, len(train_set) - 1)
        omega = train_set[omega_index]
        omega_label = train_label[omega_index]
        # 使用删除法删除已标记的点
        train_set = np.delete(train_set, omega_index, axis=0)
        train_label = np.delete(train_label, omega_index, axis=0)
        # 首先判断是否还有点可供训练
        if len(train_set) != 0:
            # 若有，计算异类点的内积
            # 由于 PEP 8: E501 的限制，下边将尽量不使用列表推导式，以防单行代码过长
            different_inner_product = []
            for idx, sample in enumerate(train_set):
                if train_label[idx] != omega_label:
                    different_inner_product.append(np.dot(sample, omega))
            # 判断是否仍有异类点
            if len(different_inner_product) != 0:
                # 计算最近异类点内积
                most_near_different_inner_product = max(different_inner_product)
                # 计算所有大于最近异类点内积（也就是比最近异类点的距离还小的点）的同类点
                in_range = []
                for idx, sample in enumerate(train_set):
                    if train_label[idx] == omega_label:
                        if np.dot(sample, omega) > most_near_different_inner_product:
                            in_range.append([idx, np.dot(sample, omega)])
                in_range = np.array(in_range)
                # 判断该范围内是否有同类点
                if len(in_range) != 0:
                    # 计算最远同类点
                    most_far_similar_inner_product = min(in_range[:, 1])
                    # 采用平均半径法
                    average = (most_far_similar_inner_product + most_near_different_inner_product) / 2
                    in_range_samples, train_set, train_label = _delete_in_range(in_range, train_set, train_label, omega)
                    model.append({'center': omega,
                                  'r': average,
                                  'label': omega_label,
                                  'count': len(in_range) + 1,
                                  'in_range_samples': in_range_samples
                                  })
                # 若没有，采用最近异类点距离的一半，也就是内积的二倍
                else:
                    model.append({'center': omega,
                                  'r': most_near_different_inner_product * 2,
                                  'label': omega_label,
                                  'count': 1,
                                  'in_range_samples': np.array(omega)
                                  })
            # 若没有，则直接寻找最远同类点
            else:
                in_range = np.array([[idx, np.dot(sample, omega)] for idx, sample in enumerate(train_set)])
                most_far_similar_inner_product = min(in_range[:, 0])
                in_range_samples, train_set, train_label = _delete_in_range(in_range, train_set, train_label, omega)
                model.append({'center': omega,
                              'r': most_far_similar_inner_product,
                              'label': omega_label,
                              'count': len(in_range) + 1,
                              'in_range_samples': in_range_samples})
        # 否则使用覆盖中心的自身内积当作其覆盖半径
        else:
            model.append(
                {'center': omega,
                 'r': np.dot(omega, omega),
                 'label': omega_label,
                 'count': 1,
                 'in_range_samples': np.array(omega)
                 })
    return model

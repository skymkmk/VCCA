import csv
import os
import re

import numpy as np
import sklearn.model_selection
import tabulate
import yaml

from src import *

if __name__ == '__main__':
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)
    K_FOLD_NUM = config["k_fold_num"]
    ROUND = config["round"]
    VCCA_MODEL_NUM = config["vcca_model_num"]
    NEIGHBOURS_NUM = config["neighbours_num"]
    results = []
    # 遍历数据集文件夹
    for root, dirs, files in os.walk("./datasets/"):
        for file in files:
            # 仅读取后缀名为 .data 的文件
            if re.match(r".*\.data", file):
                result = [file.replace(".data", '')]
                dataset = []
                # 由于 .data 文件是 CSV 格式，故采用 CSV 模块读取
                with open(os.path.join(root, file), 'r') as f:
                    reader = csv.reader(f)
                    for i in reader:
                        # 防止出现空行
                        if len(i) != 0:
                            dataset.append(i)
                # 预处理，包含数据集与标签的分离、缺失项填充、归一化与升维
                dataset, label = pre_process(dataset, NEIGHBOURS_NUM)
                cca_total_correct_identified = 0
                cca_total_cant_identified = 0
                vcca_total_correct_identified = 0
                vcca_total_cant_identified = 0
                # 采用十折交叉验证
                kf = sklearn.model_selection.KFold(n_splits=K_FOLD_NUM, shuffle=True)
                epoch = 1
                for i in range(ROUND):
                    for train_idx, test_idx in kf.split(dataset, label):
                        print(file.replace(".data", ''), "Epoch", epoch)
                        epoch += 1
                        model = []
                        train_set, train_label = np.array(dataset[train_idx]), label[train_idx]
                        test_set, test_label = np.array(dataset[test_idx]), label[test_idx]
                        # 测试 CCA
                        model_cca = cca(train_set, train_label)
                        cca_correct_identified, cca_cant_identified = test_cca(model_cca, test_set, test_label)
                        print("CCA: Rate:" + str(round(cca_correct_identified / len(test_idx) * 100, 2)) + '%,',
                              end=' ')
                        print("can't identified:", cca_cant_identified)
                        cca_total_correct_identified += cca_correct_identified
                        cca_total_cant_identified += cca_cant_identified
                        model_vcca = vcca(train_set, train_label, VCCA_MODEL_NUM)
                        vcca_correct_identified, vcca_cant_identified = test_vcca(model_vcca, test_set, test_label)
                        print("VCCA: Rate:" + str(round(vcca_correct_identified / len(test_idx) * 100, 2)) + '%',
                              end=' ')
                        print(", can't identified:", vcca_cant_identified)
                        vcca_total_correct_identified += vcca_correct_identified
                        vcca_total_cant_identified += vcca_cant_identified
                result.append(str(round(cca_total_correct_identified / (K_FOLD_NUM * ROUND) / (len(dataset) /
                                                                                               K_FOLD_NUM)
                                        * 100, 2)) + '%')
                result.append(cca_total_cant_identified / (K_FOLD_NUM * ROUND))
                result.append(str(round(vcca_total_correct_identified / (K_FOLD_NUM * ROUND) / (len(dataset) /
                                                                                                K_FOLD_NUM)
                                        * 100, 2)) + '%')
                result.append(vcca_total_cant_identified / (K_FOLD_NUM * ROUND))
                results.append(result)
    print(tabulate.tabulate(results, headers=["Dataset", "CCA-Avg rate", "CCA-Avg can't identified", "VCCA-Avg rate",
                                              "VCCA-Avg can't identified"], tablefmt="pretty"))

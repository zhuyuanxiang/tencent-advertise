# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   load_data.py
@Version    :   v0.1
@Time       :   2020-08-15 11:59
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import numpy as np
import pandas as pd

import config
import tools
from tensorflow import keras

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
from show_data import show_example_data, show_original_x_data

if __name__ == '__main__':
    # 运行结束的提醒
    tools.beep_end()


def load_data_set(file_path):
    data_type = "训练数据集"
    print('-' * 5 + ' ' * 3 + "加载{0}".format(data_type) + ' ' * 3 + '-' * 5)

    x_train = np.load(file_path + 'x_train.npy', allow_pickle=True)
    y_train = np.load(file_path + 'y_train.npy', allow_pickle=True)
    show_example_data(x_train, y_train, data_type)
    print(data_type + "加载成功。")

    data_type = "测试数据集"
    print('-' * 5 + ' ' * 3 + "加载{0}".format(data_type) + ' ' * 3 + '-' * 5)
    x_test = np.load(file_path + 'x_test.npy', allow_pickle=True)
    y_test = np.load(file_path + 'y_test.npy', allow_pickle=True)
    show_example_data(x_test, y_test, data_type)
    print(data_type + "加载成功。")

    return x_train, y_train, x_test, y_test


def load_word2vec_file(file_name, field_list):
    # 载入数据需要的变量
    print('-' * 5 + "   加载数据集:{0}   ".format(file_name) + '-' * 5)
    # 「CSV」文件字段名称
    df = pd.read_csv(file_name, dtype=int)
    # --------------------------------------------------
    # 没有在数据库中处理索引，是因为尽量不在数据库中修正原始数据，除非是不得不变更的数据，这样子业务逻辑清楚
    # user_id_inc:      X_csv[:,0]
    # creative_id_inc:  X_csv[:,1]
    # 索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的，因此字段的偏移量为 -1
    x_csv = df[field_list].values - 1
    # 'creative_id_inc' 字段的偏移量为 3，是因为需要保留 {0, 1, 2}：0 表示 “padding”（填充），1 表示 “unknown”（未知词），2 表示 “start”（序列开始）
    x_csv[:, 1] = x_csv[:, 1] + 3
    print("数据加载完成。")
    show_original_x_data(x_csv)
    return x_csv

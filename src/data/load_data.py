# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
--------------------------------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   load_data.py
@Version    :   v0.1
@Time       :   2020-06-07 15:41
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :   数据载入模块
@理解：
"""
import numpy as np
import pandas as pd

import config
from src.data.show_data import show_data_result
from src.data.show_data import show_example_data
from tools import get_w2v_file_name
from tools import show_title


# --------------------------------------------------
def load_original_data():
    """
      从 csv 文件中载入原始数据
      :return: x_csv, y_csv
    """
    # 「CSV」文件字段名称
    df = pd.read_csv(config.load_file_name, dtype=int)
    # --------------------------------------------------
    # 没有在数据库中处理索引，是因为尽量不在数据库中修正原始数据，除非是不得不变更的数据，这样子业务逻辑清楚
    # user_id:          X_csv[:,0]      需要减 1
    # creative_id:      X_csv[:,1]
    # time_id:          X_csv[:,2]      需要减 1
    # click_times:      X_csv[:,3]      数据已经是值，不是索引，不需要减 1
    # product_category: X_csv[:,4]      0 保留不使用，因此不需要减 1
    # 索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的，因此字段的偏移量为 -1
    x_csv = df[config.field_list].values
    # 'creative_id_inc' 字段的偏移量为 3，是因为需要保留 {0, 1, 2}：0 表示 “padding”（填充），1 表示 “unknown”（未知词），2 表示 “start”（序列开始）
    # TODO：建议 SQL 导出数据前就已经将数据序号改成从0开始，将 creative_id 的序号保留 0，1，2
    x_csv[:, 0] = x_csv[:, 0] - 1  # user_id
    x_csv[:, 1] = x_csv[:, 1] + (3 - 1)
    x_csv[:, 2] = x_csv[:, 2] - 1  # time_id
    # --------------------------------------------------
    # 目标数据处理：目标字段的偏移量是 -1，是因为索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的
    # 既加载了 'age'，也加载了 'gender'
    y_csv = df[config.label_name].values - 1
    print("数据加载完成。")
    show_example_data(x_csv, y_csv, '加载数据')
    return x_csv, y_csv


# --------------------------------------------------
def load_word2vec_weights():
    file_name = get_w2v_file_name()
    show_title("加载 word2vec 模型 {0}".format(file_name))
    embedding_weights = np.load(file_name + '.npy', allow_pickle=True)
    print("Word2Vec 模型加载完成。")
    return embedding_weights


# --------------------------------------------------
def load_model_data(file_name, file_path=config.data_file_path, data_type='原始数据集'):
    """
    加载训练使用的数据集
    :param file_path:
    :param file_name:
    :param data_type:
    :return:
    """
    print("加载{0}：{1}.npy --> ".format(data_type, file_path + file_name), end='')
    data = np.load(file_path + file_name + '.npy', allow_pickle=True)
    print(data_type + ':{}条数据 -->加载成功！'.format(len(data)))
    show_data_result(data, data_type)
    return data


def load_train_val_data():
    from config import train_val_data_type, x_train_val_file_name, y_train_val_file_name
    x_train_val = load_model_data(x_train_val_file_name, data_type=train_val_data_type)
    y_train_val = load_model_data(y_train_val_file_name, data_type=train_val_data_type)
    return x_train_val, y_train_val


def load_train_data():
    from config import train_data_type, x_train_file_name, y_train_file_name
    x_train = load_model_data(x_train_file_name, data_type=train_data_type)
    y_train = load_model_data(y_train_file_name, data_type=train_data_type)
    return x_train, y_train


def load_base_data():
    from config import base_data_type, x_data_file_name, y_data_file_name
    x_data = load_model_data(x_data_file_name, data_type=base_data_type)
    y_data = load_model_data(y_data_file_name, data_type=base_data_type)
    return x_data, y_data


def load_val_data():
    from config import val_data_type, x_val_file_name, y_val_file_name
    x_val = load_model_data(x_val_file_name, data_type=val_data_type)
    y_val = load_model_data(y_val_file_name, data_type=val_data_type)
    return x_val, y_val


def load_test_data():
    from config import test_data_type, x_test_file_name, y_test_file_name
    x_test = load_model_data(x_test_file_name, data_type=test_data_type)
    y_test = load_model_data(y_test_file_name, data_type=test_data_type)
    return x_test, y_test

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
import config
import numpy as np
import pandas as pd
from show_data import show_example_data, show_word2vec_data, show_original_x_data, show_original_data, show_data_result


# ----------------------------------------------------------------------
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


def load_word2vec_weights():
    from gensim.models import KeyedVectors
    from config import embedding_size, embedding_window, creative_id_window
    file_name = config.model_w2v_path + 'embedding_{0}_{1}.kv'.format(embedding_size, embedding_window)
    print('-' * 5 + ' ' * 3 + "加载 word2vec 模型 {0}".format(file_name) + ' ' * 3 + '-' * 5)
    model_w2v = KeyedVectors.load(file_name)
    embedding_weights = np.empty((creative_id_window, embedding_size))
    # 需要将训练的单词(word) 与 数组的序号(ord(word))对应
    for word, index in model_w2v.vocab.items():
        try:
            embedding_weights[ord(word), :] = model_w2v[word]
        except KeyError:
            pass
    pass
    # 0 是占位符，因此不记入模型的数据
    # 将 0 的权重大小设置为 0.5，效果并不好
    embedding_weights[0, :] = np.zeros(embedding_size)
    print("Word2Vec 模型加载完成。")
    return embedding_weights


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


def load_data(file_name, data_type='原始数据集'):
    print("加载{0}：{1}.npy --> ".format(data_type, file_name), end='')
    data = np.load(file_name + '.npy', allow_pickle=True)
    print(data_type + ':{}条数据 -->加载成功！'.format(len(data)))
    show_data_result(data, data_type)
    return data

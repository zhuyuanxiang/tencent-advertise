# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
--------------------------------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   save_data.py
@Version    :   v0.1
@Time       :   2020-06-07 15:41
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :   数据保存模块
@理解：
"""

import numpy as np
import pickle
import config

from config import data_file_path, label_name, creative_id_window
from show_data import show_example_data, show_original_data, show_word2vec_data


def save_word2vec_data(x_creative_id, creative_id_window, file_path):
    file_name = file_path + 'creative_id_{0}'.format(creative_id_window)
    print('-' * 5 + "   保存数据集:{0}   ".format(file_name) + '-' * 5)
    with open(file_name, 'wb') as fname:
        pickle.dump(x_creative_id, fname, -1)
    print("数据保存成功")


def save_word2vec_weights(model_w2v):
    file_name = config.model_w2v_path + 'embedding_{0}_{1}.kv'.format(config.embedding_size, config.embedding_window)
    print('-' * 5 + ' ' * 3 + "保存 word2vec 模型 {0}".format(file_name) + ' ' * 3 + '-' * 5)
    model_w2v.wv.save(file_name)
    print("Word2Vec 模型保存完成。")


# ----------------------------------------------------------------------
# 保存用于训练和测试模型时使用的数据集，缩短模型调优时载入数据的时间，固定模型训练的数据，方便模型对比
def save_data_set(x_train, y_train, x_test, y_test):
    data_type = "训练数据集"
    print('-' * 5 + "   保存{0}：{1}   ".format(data_type, len(y_train)) + '-' * 5)
    show_example_data(x_train, y_train, data_type)
    np.save(data_file_path + 'x_train', x_train)
    np.save(data_file_path + 'y_train', y_train)
    print(data_type + "保存成功。")

    data_type = "测试数据集"
    print('-' * 5 + "   保存{0}：{1}   ".format(data_type, len(y_train)) + '-' * 5)
    show_example_data(x_test, y_test, data_type)
    np.save(data_file_path + 'x_test', x_test)
    np.save(data_file_path + 'y_test', y_test)
    print(data_type + "保存成功。")


# ----------------------------------------------------------------------
def save_data(data, file_name, data_type='原始数据'):
    if isinstance(data[0], list) and isinstance(data[0][0], str):
        show_word2vec_data(data, data_type)
    else:
        show_original_data(data, data_type)
    print("保存{0}：{1}条数据 --> ".format(data_type, len(data)), end='')
    np.save(file_name, data)
    print(data_type + ':{}.npy --> 保存成功！'.format(file_name))

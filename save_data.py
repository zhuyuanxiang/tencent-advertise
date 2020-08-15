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

import config
from show_data import show_data_result
from tools import show_title


# ----------------------------------------------------------------------
def save_word2vec_weights(model_w2v):
    """保存 word2vec 训练的权重
    :param model_w2v:
    :return:
    """
    file_name = get_w2v_file_name()
    show_title("保存 word2vec 模型 {0}".format(file_name))
    model_w2v.wv.save(file_name)
    print("Word2Vec 模型保存完成。")


def get_w2v_file_name():
    file_name = config.model_w2v_path + config.model_file_prefix + 'w2v.kv'.format(
            config.embedding_size, config.embedding_window)
    return file_name


# ----------------------------------------------------------------------
def save_model_data(data, file_name, data_type='原始数据'):
    """
    保存训练模型使用的数据
    :param data:
    :param file_name:
    :param data_type:
    :return:
    """
    show_data_result(data, data_type)
    print("保存{0}：{1}条数据 --> ".format(data_type, len(data)), end='')
    np.save(file_name, data)
    print(data_type + ':{}.npy --> 保存成功！'.format(file_name))

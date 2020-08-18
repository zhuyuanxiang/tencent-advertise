# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
--------------------------------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   data.py
@Version    :   v0.1
@Time       :   2020-06-07 15:41
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :   数据保存模块
@理解：
"""

import numpy as np

import config
from src.data.show_data import show_data_result


# ----------------------------------------------------------------------
def save_word2vec_weights(model_w2v):
    """保存 word2vec 训练的权重
    :param model_w2v:
    :return:
    """
    from tools import show_title, get_w2v_file_name
    file_name = get_w2v_file_name()
    show_title(f"保存 word2vec 模型 {file_name}")
    # 初始化嵌入式模型权重矩阵；0 是占位符，因此不记入模型的数据；补：将 0 的权重大小设置为 0.5，效果并不好
    embedding_weights = np.zeros((config.creative_id_window, config.embedding_size))
    # 需要将训练的单词(word) 与 数组的序号(ord(word))对应
    for word, index in model_w2v.vocab.items():
        try:
            embedding_weights[ord(word), :] = model_w2v[word]
        except KeyError:
            print(f"错误的键值{word}")

    model_w2v.save(file_name)
    np.save(file_name, embedding_weights)
    print("Word2Vec 模型保存完成。")


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
    print(f'{data_type}:{file_name}.npy --> 保存成功！')

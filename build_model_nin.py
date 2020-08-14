# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model_nin.py
@Version    :   v0.1
@Time       :   2020-07-25 9:40
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# common imports
from build_model import build_single_input, build_single_output
from config import embedding_size
from keras import Sequential
from keras.layers import Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D


# ----------------------------------------------------------------------
def build_nin():
    """
    网络中的网络，训练这个数据效果不好，可能是参数不合适
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(nin_block(embedding_size * 2, 3, 2, 'valid'))
    model.add(MaxPooling1D())
    model.add(nin_block(embedding_size * 3, 2, 1, 'same'))
    model.add(MaxPooling1D())
    model.add(nin_block(embedding_size * 4, 2, 1, 'same'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(nin_block(1, 2, 1, 'same'))
    model.add(GlobalAveragePooling1D())
    return build_single_output(model)


def nin_block(num_channels, kernel_size, strides, padding):
    blk = Sequential()
    blk.add(Conv1D(num_channels, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu'))
    blk.add(Conv1D(num_channels, kernel_size=1, activation='relu'))
    blk.add(Conv1D(num_channels, kernel_size=1, activation='relu'))
    return blk

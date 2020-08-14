# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model_RNN.py
@Version    :   v0.1
@Time       :   2020-08-14 11:26
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   循环神经网络
@理解：
"""
# common imports
from keras.layers import GRU, Bidirectional, LSTM, Conv1D
from keras.regularizers import l2

import config
import tools
from tensorflow import keras

from build_model import build_single_input, build_single_output
from config import embedding_size


# ----------------------------------------------------------------------
def build_gru():
    """
    门控循环单元
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(GRU(embedding_size, dropout=0.2, recurrent_dropout=0.2))
    # model.add(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5))
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_lstm():
    """
    长短期记忆
    :param model:
    :return:
    """
    model = build_single_input()
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_bidirectional_lstm():
    """
    双向循环神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(Bidirectional(LSTM(embedding_size, dropout=0.2, recurrent_dropout=0.2)))
    return build_single_output(model)


def build_conv1d_lstm():
    model = build_single_input()

    model.add(Conv1D(embedding_size, 5, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size, 5, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(LSTM(embedding_size, dropout=0.5, recurrent_dropout=0.5))
    return build_single_output(model)

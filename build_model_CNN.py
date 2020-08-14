# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model_CNN.py
@Version    :   v0.1
@Time       :   2020-08-14 11:23
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   卷积神经网络模型
@理解：
"""
# common imports
from build_model import build_single_input, build_single_output
from config import embedding_size
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, GlobalMaxPooling1D
from keras.regularizers import l2


def build_conv1d_mlp():
    model = build_single_input()
    model.add(Conv1D(embedding_size * 2, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 2, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(embedding_size * 4, activation='relu', kernel_regularizer=l2(0.001)))
    return build_single_output(model)


def build_conv1d() -> object:
    model = build_single_input()
    model.add(Conv1D(embedding_size * 2, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 2, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Conv1D(embedding_size * 4, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 4, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Conv1D(embedding_size * 8, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 8, 2, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(MaxPooling1D())
    # model.add(Conv1D(embedding_size * 16, 2, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Conv1D(embedding_size * 16, 2, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(MaxPooling1
    # model.add(Flatten())
    # model.add(Conv1D(embedding_size * 2, 3, strides=2, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Conv1D(embedding_size * 4, 3, strides=2, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Conv1D(embedding_size * 8, 3, strides=2, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Conv1D(embedding_size * 16, 3, strides=2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    return build_single_output(model)


def build_global_max_pooling1d():
    model = build_single_input()
    model.add(GlobalMaxPooling1D())
    return build_single_output(model)


def build_le_net():
    """
    卷积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(Conv1D(embedding_size * 2, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Conv1D(embedding_size * 3, 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(embedding_size * 3, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    return build_single_output(model)


def build_alex_net():
    """
    深度总面积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(Conv1D(embedding_size * 2, kernel_size=2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 2, kernel_size=2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Conv1D(embedding_size * 4, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 4, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Conv1D(embedding_size * 8, kernel_size=2, padding='valid', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 8, kernel_size=2, padding='valid', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 4, kernel_size=2, padding='valid', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(embedding_size * 8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_size * 8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    return build_single_output(model)


def build_dense_net():
    """
    稠密连接网络
    :param model:
    :return:
    """
    model = build_single_input()
    return build_single_output(model)


def build_rcnn():
    """
    区域卷积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    return build_single_output(model)


def build_text_cnn():
    """
    文本卷积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    return build_single_output(model)


def build_fcn():
    """
    全卷积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    return build_single_output(model)

# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model.py
@Version    :   v0.1
@Time       :   2020-07-17 8:32
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   建造各种模型
@理解：
"""
# common imports
import os
import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import winsound
from Cython.Plex import Seq
from keras import Sequential, optimizers, losses, metrics
from keras.layers import Embedding, Flatten, Dropout, Dense, Conv1D, GlobalMaxPooling1D, GRU, LSTM, Bidirectional, MaxPooling1D
from keras.regularizers import l2
from tensorflow import keras

# ----------------------------------------------------------------------
from config import creative_id_window, embedding_size, max_len, label_name, model_type, learning_rate
from load_data import load_word2vec_weights

plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ >= "1.18.1"


def build_single_input():
    model = Sequential()
    # mask_zero 在 MaxPooling 层中不能支持
    model.add(Embedding(creative_id_window, embedding_size, input_length=max_len, weights=[load_word2vec_weights()], trainable=False))
    return model


def build_single_output(model: keras.Sequential):
    if label_name == "age":
        model.add(Dense(embedding_size * 10, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(10, activation='softmax', kernel_regularizer=l2(0.001)))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(
            optimizer=optimizers.RMSprop(learning_rate),
            loss=losses.sparse_categorical_crossentropy,
            metrics=[metrics.sparse_categorical_accuracy]
        )
    elif label_name == 'gender':
        # model.add(Dense(embedding_size, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(
            # optimizer=optimizers.RMSprop(lr=RMSProp_lr),
            optimizer=optimizers.Adam(learning_rate),
            loss=losses.binary_crossentropy,
            metrics=[metrics.binary_accuracy])
    else:
        raise Exception("错误的标签类型！")
    return model


# ----------------------------------------------------------------------
def build_mlp():
    model = build_single_input()
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(embedding_size * max_len // 4, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_conv1d_mlp():
    model = build_single_input()
    model.add(Conv1D(64, 7, strides=2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(128, 7, strides=2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(embedding_size * max_len // 4, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_conv1d():
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
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_global_max_pooling1d():
    model = build_single_input()
    model.add(GlobalMaxPooling1D())
    model.add(build_single_output(model))
    return build_single_output(model)


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
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_lstm():
    """
    长短期记忆
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_conv1d_lstm():
    model = build_single_input()

    model.add(Conv1D(32, 5, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(32, 5, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(LSTM(16, dropout=0.5, recurrent_dropout=0.5))
    model.add(build_single_output(model))
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
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
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
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
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
    model.add(Conv1D(embedding_size * 3, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 3, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Conv1D(embedding_size * 4, kernel_size=2, padding='valid', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 4, kernel_size=2, padding='valid', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(embedding_size * 3, kernel_size=2, padding='valid', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(embedding_size * embedding_size // 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(embedding_size * embedding_size // 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(build_single_output(model))
    return build_single_output(model)


# ----------------------------------------------------------------------
def build_vgg():
    """
    使用重复元素的网络
    :param model:
    :return:
    """
    model = build_single_input()
    conv_arch = ((2, 64), (2, 128), (2, 256), (2, 256))
    for (num_convs, num_channels) in conv_arch:
        model.add(vgg_block(num_convs, num_channels))
    model.add(Flatten())
    model.add(Dense(embedding_size * 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_size * 2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    return build_single_output(model)


def vgg_block(num_convs, num_channels):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(Conv1D(num_channels, kernel_size=2, padding='same', activation='relu'))
    blk.add(MaxPooling1D())
    return blk


# ----------------------------------------------------------------------
def build_nin():
    """
    网络中的网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(nin_block(64, 3, 2, 'valid'))
    model.add(MaxPooling1D())
    model.add(nin_block(128, 3, 1, 'same'))
    model.add(MaxPooling1D())
    model.add(nin_block(64, 2, 1, 'same'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(nin_block(1, 2, 1, 'same'))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(build_single_output(model))
    return model


def nin_block(num_channels, kernel_size, strides, padding):
    blk = Sequential()
    blk.add(Conv1D(num_channels, kernel_size, strides, padding, activation='relu'))
    blk.add(Conv1D(num_channels, kernel_size=1, activation='relu'))
    blk.add(Conv1D(num_channels, kernel_size=1, activation='relu'))
    return blk


# ----------------------------------------------------------------------
def build_google_net():
    """
    含并行连结的网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(build_single_output(model))
    return model


# ----------------------------------------------------------------------
def build_res_net():
    """
    残差网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(build_single_output(model))
    return model


# ----------------------------------------------------------------------
def build_dense_net():
    """
    稠密连接网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(build_single_output(model))
    return model


# ----------------------------------------------------------------------
def build_rcnn():
    """
    区域卷积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(build_single_output(model))
    return model


# ----------------------------------------------------------------------
def build_text_cnn():
    """
    文本卷积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(build_single_output(model))
    return model


# ----------------------------------------------------------------------
def build_fcn():
    """
    全卷积神经网络
    :param model:
    :return:
    """
    model = build_single_input()
    model.add(build_single_output(model))
    return model


# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

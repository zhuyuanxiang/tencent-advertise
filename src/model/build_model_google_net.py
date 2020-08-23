# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model_google_net.py
@Version    :   v0.1
@Time       :   2020-07-25 9:36
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# common imports
import os
import numpy as np
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, concatenate, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.regularizers import l2

# ----------------------------------------------------------------------
from src.model.build_model import build_single_output_api, build_single_model_api, build_creative_id_input, build_embedded_creative_id
from src.base.config import embedding_size

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)


# ----------------------------------------------------------------------
def build_google_net():
    """
    含并行连结的网络，没有串联多个模块是因为显卡无法高效训练所有参数
    :return:
    """
    model_input = [build_creative_id_input()]
    x0 = build_embedded_creative_id(model_input)

    # x1 = build_inception_mp(x0, embedding_size * 1, 1)
    # x1 = Dropout(0.2)(x1)
    # x1 = Dense(embedding_size * 2, activation='relu', kernel_regularizer=l2(0.001))(x1)

    # x2 = build_inception_mp(x1, embedding_size * 2, 2)
    # x2 = Dropout(0.2)(x2)
    # x2 = Dense(embedding_size * 4, activation='relu', kernel_regularizer=l2(0.001))(x2)

    # x3 = build_inception_mp(x2, embedding_size * 4, 3)
    # x3 = Dropout(0.2)(x3)
    # x3 = Dense(embedding_size * 8, activation='relu', kernel_regularizer=l2(0.001))(x3)

    # x_output = concatenate([build_inception_ga(x0, embedding_size, 4), build_inception_gm(x0, embedding_size, 5)])
    x_output = build_inception_gm(x0, embedding_size, 4)
    x_output = Dropout(0.2)(x_output)
    x_output = Dense(embedding_size, activation='relu', kernel_regularizer=l2(0.001))(x_output)
    model_output = build_single_output_api(x_output)
    return build_single_model_api(model_input, model_output)


# ----------------------------------------------------------------------
def build_inception_mp(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x1_conv1d'.format(inception_num))(inception_input)
    x1 = MaxPooling1D(name='incp{}_x1_mp'.format(inception_num))(x1)

    x2 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x2_conv1d'.format(inception_num))(inception_input)
    x2 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x2_1'.format(inception_num))(x2)
    x2 = MaxPooling1D(name='incp{}_x2_mp'.format(inception_num))(x2)

    x3 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x3_conv1d_1'.format(inception_num))(inception_input)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_conv1d_2'.format(inception_num))(x3)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_conv1d_3'.format(inception_num))(x3)
    x3 = MaxPooling1D(name='incp{}_x3_mp'.format(inception_num))(x3)

    x4 = MaxPooling1D(name='incp{}_x4_mp'.format(inception_num))(inception_input)
    x4 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x4_conv1d'.format(inception_num))(x4)

    inception_output = concatenate([x1, x2, x3, x4], axis=-1)
    return inception_output


# ----------------------------------------------------------------------
def build_inception_ap(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x1_conv1d'.format(inception_num))(inception_input)
    x1 = AveragePooling1D(name='incp{}_x1_ap'.format(inception_num))(x1)

    x2 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x2_conv1d'.format(inception_num))(inception_input)
    x2 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x2_1'.format(inception_num))(x2)
    x2 = AveragePooling1D(name='incp{}_x2_ap'.format(inception_num))(x2)

    x3 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x3_conv1d'.format(inception_num))(inception_input)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_1'.format(inception_num))(x3)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_2'.format(inception_num))(x3)
    x3 = AveragePooling1D(name='incp{}_x3_ap'.format(inception_num))(x3)

    x4 = AveragePooling1D(name='incp{}_x4_ap'.format(inception_num))(inception_input)
    x4 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x4_conv1d'.format(inception_num))(x4)

    inception_output = concatenate([x1, x2, x3, x4], axis=-1)
    return inception_output


# ----------------------------------------------------------------------
def build_inception_gm(inception_input, num_channels, inception_num):
    """
    TODO: 使用 model 封装模型，使模型更加易于理解
    :param inception_input:
    :param num_channels:
    :param inception_num:
    :return:
    """
    x1 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x1_conv1d'.format(inception_num))(inception_input)
    x1 = GlobalMaxPooling1D(name='incp{}_x1_gm'.format(inception_num))(x1)

    x2 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x2_conv1d'.format(inception_num))(inception_input)
    x2 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x2_1'.format(inception_num))(x2)
    x2 = GlobalMaxPooling1D(name='incp{}_x2_gm'.format(inception_num))(x2)

    x3 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x3_conv1d'.format(inception_num))(inception_input)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_1'.format(inception_num))(x3)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_2'.format(inception_num))(x3)
    x3 = GlobalMaxPooling1D(name='incp{}_x3_gm'.format(inception_num))(x3)

    x4 = MaxPooling1D(name='incp{}_x4_mp'.format(inception_num))(inception_input)
    x4 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x4_conv1d'.format(inception_num))(x4)
    x4 = GlobalMaxPooling1D(name='incp{}_x4_gm'.format(inception_num))(x4)

    inception_output = concatenate([x1, x2, x3, x4], axis=-1)
    return inception_output


# ----------------------------------------------------------------------
def build_inception_ga(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x1_conv1d'.format(inception_num))(inception_input)
    x1 = GlobalAveragePooling1D(name='incp{}_x1_ga'.format(inception_num))(x1)

    x2 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x2_conv1d'.format(inception_num))(inception_input)
    x2 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x2_1'.format(inception_num))(x2)
    x2 = GlobalAveragePooling1D(name='incp{}_x2_ga'.format(inception_num))(x2)

    x3 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x3_conv1d'.format(inception_num))(inception_input)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_1'.format(inception_num))(x3)
    x3 = Conv1D(num_channels * 2, 2, padding='same', activation='relu', name='incp{}_x3_2'.format(inception_num))(x3)
    x3 = GlobalAveragePooling1D(name='incp{}_x3_ga'.format(inception_num))(x3)

    x4 = MaxPooling1D(name='incp{}_x4_mp'.format(inception_num))(inception_input)
    x4 = Conv1D(num_channels * 2, 1, activation='relu', name='incp{}_x4_conv1d'.format(inception_num))(x4)
    x4 = GlobalAveragePooling1D(name='incp{}_x4_ga'.format(inception_num))(x4)

    inception_output = concatenate([x1, x2, x3, x4], axis=-1)
    return inception_output

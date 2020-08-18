# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model_res_net.py
@Version    :   v0.1
@Time       :   2020-07-25 9:52
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# common imports
from keras.layers import Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dropout, Dense, GlobalMaxPooling1D, add, concatenate, MaxPooling1D, \
    AveragePooling1D
from keras.regularizers import l2
from src.model.build_model import build_single_output_api, build_single_model_api, build_creative_id_input, build_embedded_creative_id
from config import embedding_size


# ----------------------------------------------------------------------
def build_res_net():
    """
    残差网络
    :param model:
    :return:
    """
    model_input = [build_creative_id_input()]
    x0 = build_embedded_creative_id(model_input)
    # x1 = build_residual_mp(x0, embedding_size * 2, 1)
    # x1 = build_residual_mp_2(x0, embedding_size * 2, 1)
    # x2 = build_residual_mp_3(x1, embedding_size * 3, 2)
    # x3 = build_residual_mp_3(x2, embedding_size * 4, 3)
    # x_output = build_residual_gm(x1, embedding_size * 3, 4)
    # x_output = build_residual_ga(x1, embedding_size * 3, 4)
    # x_output = concatenate([
    #     build_residual_gm(x0, embedding_size * 2, 4),
    #     build_residual_ga(x0, embedding_size * 2, 5)], axis=-1)
    x_output = build_residual(x0, embedding_size * 2, 3)
    x_output = build_residual(x_output, embedding_size * 4, 4)
    x_output = concatenate([
        GlobalMaxPooling1D()(x_output),
        GlobalAveragePooling1D()(x_output)
    ], axis=-1)
    x_output = Dropout(0.2)(x_output)
    x_output = Dense(embedding_size, activation='relu', kernel_regularizer=l2(0.001))(x_output)

    model_output = build_single_output_api(x_output)
    return build_single_model_api(model_input, model_output)


# ----------------------------------------------------------------------
def build_residual_mp(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels, 1, name='res{}_x1_conv1d'.format(inception_num))(inception_input)

    x2 = Conv1D(num_channels, 1, name='res{}_x2_conv1d_1'.format(inception_num))(inception_input)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(num_channels, 2, padding='same', name='res{}_x2_conv1d_2'.format(inception_num))(x2)

    x3 = Conv1D(num_channels, 1, name='res{}_x3_conv1d_1'.format(inception_num))(inception_input)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv1D(num_channels, 2, padding='same', name='res{}_x3_conv1d_2'.format(inception_num))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv1D(num_channels, 2, padding='same', name='res{}_x3_conv1d_3'.format(inception_num))(x3)

    inception_output = add([x1, x2, x3])
    inception_output = BatchNormalization()(inception_output)
    inception_output = Activation('relu')(inception_output)
    inception_output = MaxPooling1D()(inception_output)
    return inception_output


# ----------------------------------------------------------------------
def build_residual_mp_2(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels, 1, name='res{}_x1_conv1d'.format(inception_num))(inception_input)

    x2 = Conv1D(num_channels, 1, name='res{}_x2_conv1d_1'.format(inception_num))(inception_input)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(num_channels, 2, padding='same', name='res{}_x2_conv1d_2'.format(inception_num))(x2)

    inception_output = add([x1, x2])
    inception_output = BatchNormalization()(inception_output)
    inception_output = Activation('relu')(inception_output)
    inception_output = MaxPooling1D()(inception_output)
    return inception_output


# ----------------------------------------------------------------------
def build_residual_mp_3(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels, 1, name='res{}_x1_conv1d'.format(inception_num))(inception_input)

    x3 = Conv1D(num_channels, 1, name='res{}_x3_conv1d_1'.format(inception_num))(inception_input)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv1D(num_channels, 2, padding='same', name='res{}_x3_conv1d_2'.format(inception_num))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv1D(num_channels, 2, padding='same', name='res{}_x3_conv1d_3'.format(inception_num))(x3)

    inception_output = add([x1, x3])
    inception_output = BatchNormalization()(inception_output)
    inception_output = Activation('relu')(inception_output)
    inception_output = MaxPooling1D(3)(inception_output)
    return inception_output


# ----------------------------------------------------------------------
def build_residual_ap(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels, 1, name='res{}_x1_conv1d'.format(inception_num))(inception_input)

    x2 = Conv1D(num_channels, 1, name='res{}_x2_conv1d_1'.format(inception_num))(inception_input)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(num_channels, 2, padding='same', name='res{}_x2_conv1d_2'.format(inception_num))(x2)

    x3 = Conv1D(num_channels, 1, name='res{}_x3_conv1d_1'.format(inception_num))(inception_input)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv1D(num_channels, 2, padding='same', name='res{}_x3_conv1d_2'.format(inception_num))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv1D(num_channels, 2, padding='same', name='res{}_x3_conv1d_3'.format(inception_num))(x3)

    inception_output = add([x1, x2, x3])
    inception_output = BatchNormalization()(inception_output)
    inception_output = Activation('relu')(inception_output)
    inception_output = AveragePooling1D()(inception_output)
    return inception_output


# ----------------------------------------------------------------------
def build_residual_gm(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels, 1, activation='relu', name='res{}_x1_conv1d'.format(inception_num))(inception_input)

    x2 = Conv1D(num_channels, 1, activation='relu', name='res{}_x2_conv1d_1'.format(inception_num))(inception_input)
    x2 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x2_conv1d_2'.format(inception_num))(x2)

    x3 = Conv1D(num_channels, 1, activation='relu', name='res{}_x3_conv1d_1'.format(inception_num))(inception_input)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x3_conv1d_2'.format(inception_num))(x3)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x3_conv1d_3'.format(inception_num))(x3)

    inception_output = add([x1, x2, x3])
    inception_output = GlobalMaxPooling1D()(inception_output)
    return inception_output


# ----------------------------------------------------------------------
def build_residual_ga(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels, 1, activation='relu', name='res{}_x1_conv1d'.format(inception_num))(inception_input)

    x2 = Conv1D(num_channels, 1, activation='relu', name='res{}_x2_conv1d_1'.format(inception_num))(inception_input)
    x2 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x2_conv1d_2'.format(inception_num))(x2)

    x3 = Conv1D(num_channels, 1, activation='relu', name='res{}_x3_conv1d_1'.format(inception_num))(inception_input)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x3_conv1d_2'.format(inception_num))(x3)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x3_conv1d_3'.format(inception_num))(x3)

    inception_output = add([x1, x2, x3])
    inception_output = GlobalAveragePooling1D()(inception_output)
    return inception_output


# ----------------------------------------------------------------------
def build_residual(inception_input, num_channels, inception_num):
    x1 = Conv1D(num_channels, 1, activation='relu', name='res{}_x1_conv1d'.format(inception_num))(inception_input)

    x2 = Conv1D(num_channels, 1, activation='relu', name='res{}_x2_conv1d_1'.format(inception_num))(inception_input)
    x2 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x2_conv1d_2'.format(inception_num))(x2)

    x3 = Conv1D(num_channels, 1, activation='relu', name='res{}_x3_conv1d_1'.format(inception_num))(inception_input)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x3_conv1d_2'.format(inception_num))(x3)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name='res{}_x3_conv1d_3'.format(inception_num))(x3)

    inception_output = add([x1, x2, x3])
    return inception_output

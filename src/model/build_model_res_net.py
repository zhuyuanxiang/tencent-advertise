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
import tensorflow as tf
from keras import activations
from keras import layers
from keras.layers import Activation
from keras.layers import add
from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import MaxPooling1D
from keras.regularizers import l2

from src.base.config import embedding_size
from src.model.build_model import build_creative_id_input
from src.model.build_model import build_embedded_creative_id
from src.model.build_model import build_single_model_api
from src.model.build_model import build_single_output_api


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
    x_output = build_residual(embedding_size * 3, x0, 3)
    x_output = build_residual(embedding_size * 6, x_output, 4)
    x_output = concatenate([
            GlobalMaxPooling1D()(x_output),
            GlobalAveragePooling1D()(x_output)
    ], axis=-1)
    x_output = Dropout(0.2)(x_output)
    x_output = Dense(embedding_size, activation='relu', kernel_regularizer=l2(0.001))(x_output)

    model_output = build_single_output_api(x_output)
    return build_single_model_api(model_input, model_output)


# ----------------------------------------------------------------------
def build_residual_mp(num_channels, inception_input, inception_num):
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
def build_residual_mp_2(num_channels, inception_input, inception_num):
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
def build_residual_mp_3(num_channels, inception_input, inception_num):
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
def build_residual_ap(num_channels, inception_input, inception_num):
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
def build_residual_gm(num_channels, inception_input, inception_num):
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
def build_residual_ga(num_channels, inception_input, inception_num):
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
def build_residual(num_channels, inception_input, inception_num):
    x1 = Conv1D(num_channels, 1, activation='relu', name=f'res{inception_num}_x1_conv1d')(inception_input)

    x2 = Conv1D(num_channels, 1, activation='relu', name=f'res{inception_num}_x2_conv1d_1')(inception_input)
    x2 = Conv1D(num_channels, 2, padding='same', activation='relu', name=f'res{inception_num}_x2_conv1d_2')(x2)

    x3 = Conv1D(num_channels, 1, activation='relu', name=f'res{inception_num}_x3_conv1d_1')(inception_input)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name=f'res{inception_num}_x3_conv1d_2')(x3)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name=f'res{inception_num}_x3_conv1d_3')(x3)

    inception_output = add([x1, x2, x3])
    return inception_output


def build_residual_share(num_channels, inception_input, inception_num):
    x1 = Conv1D(num_channels, 1, activation='relu', name=f'res{inception_num}_x1_conv1d')(inception_input)
    x2 = Conv1D(num_channels, 2, padding='same', activation='relu', name=f'res{inception_num}_x2_conv1d')(x1)
    x3 = Conv1D(num_channels, 2, padding='same', activation='relu', name=f'res{inception_num}_x3_conv1d')(x2)
    x4 = Conv1D(num_channels, 2, padding='same', activation='relu', name=f'res{inception_num}_x4_conv1d')(x3)
    inception_output = add([x1, x2, x3, x4])
    return inception_output


def build_residual_bn(num_channels, inception_input, inception_num):
    x1 = Conv1D(num_channels, 1, name=f'res_{inception_num}_x1_conv1d')(inception_input)
    x1 = BatchNormalization(name=f'res_{inception_num}_x1_bn')(x1)
    x2 = Conv1D(num_channels, 2, padding='same', name=f'res_{inception_num}_x2_conv1d')(x1)
    x2 = BatchNormalization(name=f'res_{inception_num}_x2_bn')(x2)
    x2_act = Activation('relu', name=f'res_{inception_num}_x2_act')(x2)
    x3 = Conv1D(num_channels, 2, padding='same', name=f'res_{inception_num}_x3_conv1d')(x2_act)
    x3 = BatchNormalization(name=f'res_{inception_num}_x3_bn')(x3)

    inception_output = add([x1, x2, x3])
    # inception_output = add([x1, x3])
    return Activation('relu', name=f'res{inception_num}_relu')(inception_output)


def build_residual_bn_api(num_channels, inception_input, inception_num):
    conv1 = Conv1D(num_channels, kernel_size=2, padding='same', name=f'res_{inception_num}_x1_conv1d')
    bn1 = BatchNormalization()
    conv2 = Conv1D(num_channels, kernel_size=2, padding='same', name=f'res_{inception_num}_x2_conv1d')
    bn2 = BatchNormalization()
    x2 = conv1(inception_input)
    x2 = bn1(x2)
    x2 = activations.relu(x2)
    x2 = conv2(x2)
    x2 = bn2(x2)
    conv3 = layers.Conv1D(num_channels, kernel_size=1, name=f'res_{inception_num}_x3_conv1d')
    x3 = conv3(inception_input)
    return activations.relu(x2 + x3)


class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=strides)
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3, padding='same')
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels,
                                       kernel_size=1,
                                       strides=strides)
        else:
            self.conv3 = None
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X, **kwargs):
        Y = activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return activations.relu(Y + X)

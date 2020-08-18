# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model_vgg.py
@Version    :   v0.1
@Time       :   2020-07-25 9:43
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# common imports
import os
import numpy as np
import winsound
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPooling1D
from keras.regularizers import l2

# ----------------------------------------------------------------------
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
from code.model.build_model import build_single_input, build_single_output
from config import embedding_size

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)


# to make this notebook's output stable across runs
# ----------------------------------------------------------------------
def build_vgg():
    """
    使用重复元素的网络
    :param model:
    :return:
    """
    model = build_single_input()
    conv_arch = ((2, 64), (2, 64), (2, 128), (2, 128))
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
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)

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
import os
import numpy as np
import winsound
from keras import Sequential
from keras.layers import Embedding, Dropout, Dense, Conv1D, MaxPooling1D, concatenate, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.regularizers import l2

# ----------------------------------------------------------------------
from build_model import build_single_input_api, build_single_output_api, build_single_model_api, build_single_input, build_single_output
from config import creative_id_window, embedding_size, max_len
from load_data import load_word2vec_weights

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)


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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)

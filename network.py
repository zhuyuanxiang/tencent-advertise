# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   network.py
@Version    :   v0.1
@Time       :   2020-06-07 17:31
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   网络构建模块
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
from tensorflow import keras
from tensorflow.python.keras.activations import relu, sigmoid
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Dense

# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ >= "1.18.1"


# ----------------------------------------------------------------------
def construct_mlp(creative_id_num, embedding_size, max_len):
    from tensorflow.python.keras.layers import Flatten
    print("* 构建网络")
    model = Sequential()
    model.add(Embedding(creative_id_num, embedding_size, input_length = max_len))
    model.add(Flatten())
    model.add(keras.layers.Dense(
            64, activation = keras.activations.relu,
            kernel_regularizer = keras.regularizers.l1(0.00001)
    ))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(
            32, activation = keras.activations.relu,
            kernel_regularizer = keras.regularizers.l1(0.00001)
    ))
    model.add(keras.layers.Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    print("MLP——模型构建完成！")
    return model


# ----------------------------------------------------------------------
def construct_Conv1d(creative_id_num, embedding_size, max_len):
    from tensorflow.python.keras.layers import Conv1D
    from tensorflow.python.keras.layers import GlobalMaxPooling1D, MaxPooling1D
    print("* 构建网络")
    model = Sequential()
    model.add(Embedding(creative_id_num, embedding_size, input_length = max_len))
    model.add(Conv1D(32, 3, activation = relu))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation = relu))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation = sigmoid))
    print("Conv1D——模型构建完成！")
    return model


# ----------------------------------------------------------------------
def construct_Conv1d_LSTM(creative_id_num, embedding_size, max_len):
    from tensorflow.python.keras.layers import GlobalMaxPooling1D, MaxPooling1D
    from tensorflow.python.keras.layers import Conv1D
    from tensorflow.python.keras.layers import GRU, LSTM
    print("* 构建网络")
    model = Sequential()
    model.add(Embedding(creative_id_num, embedding_size, input_length = max_len))
    model.add(Conv1D(8, 3, activation = relu))
    model.add(MaxPooling1D(2))
    model.add(LSTM(4, dropout = 0.5, recurrent_dropout = 0.5))
    model.add(Dense(1, activation = sigmoid))
    print("Conv1D+LSTM——模型构建完成！")
    return model


# ----------------------------------------------------------------------
def construct_LSTM(creative_id_num, embedding_size, max_len):
    from tensorflow.python.keras.layers import LSTM
    print("* 构建网络")
    model = Sequential()
    model.add(Embedding(creative_id_num, embedding_size, input_length = max_len))
    model.add(LSTM(32, dropout = 0.5, recurrent_dropout = 0.5))
    model.add(Dense(1, activation = sigmoid))
    print("LSTM——模型构建完成！")
    return model


# ----------------------------------------------------------------------
def construct_Bidirectional_LSTM(creative_id_num, embedding_size, max_len):
    from tensorflow.python.keras.layers import Bidirectional, LSTM
    print("* 构建网络")
    model = Sequential()
    model.add(Embedding(creative_id_num, embedding_size, input_length = max_len))
    model.add(Bidirectional(LSTM(embedding_size, dropout = 0.2, recurrent_dropout = 0.2)))
    model.add(Dense(1, activation = sigmoid))
    print("Bidirectional LSTM——模型构建完成！")
    return model


# ----------------------------------------------------------------------

if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

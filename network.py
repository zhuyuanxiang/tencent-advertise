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
from tensorflow.python.keras.layers import GlobalMaxPooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional, Conv1D, Dropout, Embedding, Flatten, LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics

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


def construct_model(creative_id_num, embedding_size, max_len, RMSProp_lr,
                    model_type = "MLP", label_name = "gender"):
    '''
        构建与编译模型
    :param creative_id_num: 字典容量
    :param embedding_size: 嵌入维度
    :param max_len: 序列长度
    :param RMSProp_lr: 学习步长
    :param model_type: 模型的类型
        MLP：多层感知机
        Conv1D：1维卷积神经网络
        GlobalMaxPooling1D：1维全局池化层
        GlobalMaxPooling1D+MLP：1维全局池化层+多层感知机
        Conv1D+LSTM：1维卷积神经网络+LSTM
        Bidirectional+LSTM：双向 LSTM
    :param label_name: 标签的类型
        age ： 根据年龄进行的多分类问题
        gender : 根据性别进行的二分类问题
    :return: 返回构建的模型
    '''
    print("* 构建网络")
    model = Sequential()
    model.add(Embedding(creative_id_num, embedding_size, input_length = max_len))
    if model_type == 'MLP':
        model.add(Flatten())
        model.add(Dense(8, activation = 'relu', kernel_regularizer = l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation = 'relu', kernel_regularizer = l2(0.001)))
        model.add(Dropout(0.5))
    elif model_type == 'Conv1d':
        model.add(Conv1D(32, 7, activation = 'relu', kernel_regularizer = l2(0.001)))
        model.add(Conv1D(32, 7, activation = 'relu', kernel_regularizer = l2(0.001)))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'GlobalMaxPooling1D':
        model.add(GlobalMaxPooling1D())
    elif model_type == 'GlobalMaxPooling1D+MLP':
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation = 'relu', kernel_regularizer = l2(0.001)))
        # model.add(Dropout(0.5))
        model.add(Dense(32, activation = 'relu', kernel_regularizer = l2(0.001)))
        # model.add(Dropout(0.5))
    elif model_type == 'Conv1d+LSTM':
        model.add(Conv1D(32, 5, activation = 'relu', kernel_regularizer = l2(0.001)))
        model.add(Conv1D(32, 5, activation = 'relu', kernel_regularizer = l2(0.001)))
        model.add(LSTM(16, dropout = 0.5, recurrent_dropout = 0.5))
    elif model_type == 'Bidirectional-LSTM':
        model.add(Bidirectional(LSTM(embedding_size, dropout = 0.2, recurrent_dropout = 0.2)))

    if label_name == "age":
        model.add(Dense(10, activation = 'softmax'))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
                      loss = losses.sparse_categorical_crossentropy,
                      metrics = [metrics.sparse_categorical_accuracy])
    elif label_name == 'gender':
        model.add(Dense(1, activation = 'sigmoid'))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
                      loss = losses.binary_crossentropy,
                      metrics = [metrics.binary_accuracy])
    else:
        raise Exception("错误的标签类型！")

    print(model.summary())
    return model


# ----------------------------------------------------------------------

if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   DL-Age-Keras.py
@Version    :   v0.1
@Time       :   2020-06-11 9:04
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
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
# 加载数据
from preprocessing import data_sequence, load_data, data_sequence_no_start

file_name = './data/train_data.csv'
X_data, y_data = load_data(file_name, label_name = 'age')

# ----------------------------------------------------------------------
# 定义全局变量
user_id_num = 50000  # 用户数
creative_id_num = 50000  # 素材数
max_len = 16

# ----------------------------------------------------------------------
# 清洗数据集
# X_doc, y_doc = data_sequence(X_data, y_data, user_id_num, creative_id_num)
X_doc, y_doc = data_sequence_no_start(X_data, y_data, user_id_num, creative_id_num)
# ----------------------------------------------------------------------
# 填充数据集
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(X_doc, maxlen = max_len, padding = 'post')
y = y_doc
# ----------------------------------------------------------------------
print("* 拆分数据集")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, stratify = y)
print("\t训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))

# ----------------------------------------------------------------------
# 构建模型
from network import construct_model

embedding_size = 128
RMSProp_lr = 6e-04
# model_type = "Bidirectional+LSTM"  # Bidirectional+LSTM：双向 LSTM
# model_type = "Conv1D"  # Conv1D：1 维卷积神经网络
# model_type = "Conv1D+LSTM"  # Conv1D+LSTM：1 维卷积神经网络 + LSTM
model_type = "GlobalMaxPooling1D"  # GlobalMaxPooling1D：1 维全局池化层
# model_type = "GlobalMaxPooling1D+MLP"  # GlobalMaxPooling1D+MLP：1 维全局池化层 + 多层感知机
# model_type = "LSTM"  # LSTM：循环神经网络
# model_type = "MLP"  # MLP：多层感知机
label_name = "age"
model = construct_model(creative_id_num, embedding_size, max_len,RMSProp_lr, model_type, label_name)
# ----------------------------------------------------------------------
print("* 训练模型")
print("训练数据 =", X_train[0], y_train[0])
epochs = 20
batch_size = 256
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
          validation_split = 0.2, use_multiprocessing = True, verbose = 2)

# # 训练全部数据，不分离出验证数据，使用前面训练给出的过拟合次数决定训练周期
# # TODO:效果不好？
# model.fit(X_train, y_train, epochs = 6, batch_size = batch_size,
#           use_multiprocessing = True, verbose = 2)

# ----------------------------------------------------------------------
results = model.evaluate(X_test, y_test, verbose = 0)
predictions = model.predict(X_test).squeeze()
print("\t模型预测-->", end = '')
print("\t损失值 = {}，精确度 = {}".format(results[0], results[1]))
print("\t前10个真实的目标数据 =", np.array(y_test[:10], dtype = int))
print("\t前10个预测的目标数据 =", np.array(np.argmax(predictions[:10], 1), dtype = int))
print("\t前10个预测的结果数据 =", )
print(predictions[:10])
print("实验报告参数")
print("user_id_number =", user_id_num)
print("creative_id_number =", creative_id_num)
print("max_len =", max_len)
print("embedding_size =", embedding_size)
print("epochs =", epochs)
print("batch_size =", batch_size)
print("RMSProp =", RMSProp_lr)

# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1000)
    plt.show()

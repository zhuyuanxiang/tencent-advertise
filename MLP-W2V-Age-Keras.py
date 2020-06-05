# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   MLP-W2V-Age-Keras.py
@Version    :   v0.1
@Time       :   2020-06-05 11:31
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
from tensorflow import keras
from tensorflow.python.keras.activations import relu, sigmoid
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding, LSTM, SimpleRNN
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.metrics import binary_accuracy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import one_hot
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
# ----------------------------------------------------------------------
print("* 加载数据集...")

# 「CSV」文件字段名称
# "time_id","user_id_inc","user_id","creative_id_inc","creative_id","click_times","age","gender"
filename = './data/train_data.csv'
df = pd.read_csv(filename)

y_data = df['gender'].values - 1  # 性别作为目标数据
# 选择需要的列作为输入数据
X_data = df[["time_id", "creative_id_inc", "user_id_inc", "click_times"]].values
# 索引减去2，是因为保留了 0 和 1 两个数， 0 表示 “padding”（填充），1 表示 “unknown”（未知词）
X_data[:, 1] = X_data[:, 1] + 1
# 1m 的数据量, user_id_num=373489, creative_id_num=203603
# [gender]
# ---------
# [user_id_num = 14000,creative_id_num = 7000, epochs = 10,embedding_size=200]
# 模型预测-->损失值 = 0.4021855681964329，精确度 = 0.8614285588264465
# ---------
# [user_id_num = 20000,creative_id_num = 11500, epochs = 7,embedding_size=200]
# creative_id_num = 11500 接近最优值
# 随着 creative_id_num 的增长，精确度越来越高，可能是提供的信息越来越多的缘故
# 模型预测-->损失值 = 0.3998131538391113，精确度 = 0.8664000034332275
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test)= 83.1%
# ---------
# [user_id_num = 20000,creative_id_num = 11500, epochs = 7, embedding_size=creative_id_num/75]
# 模型预测 -->损失值 = 0.40075085363388063，精确度 = 0.8619999885559082
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test)= 85.8%
# ---------
# [user_id_num = 20000,creative_id_num = 11500, epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.38599011869430544，精确度 = 0.8659999966621399
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test)= 83.3%
# ---------
# [user_id_num = 20000,creative_id_num = 12500, epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.3867114294528961，精确度 = 0.8646000027656555
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test)= 84.2%
# ---------
# [user_id_num = 20000,creative_id_num = 12500, epochs = 10, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.48680103874206543，精确度 = 0.8622000217437744
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test) =  85.7 %
# ---------
# NOTE:这个最优，可能是因为 91 天内读取素材数目超过6个：28012，91 天内读取素材数目超过 6 个可以提供充分的分类信息
# [user_id_num =28000,creative_id_num = int(user_id_num * 2 / 3), epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.40562467643192834，精确度 = 0.8788571357727051
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  75.71428571428571 %
# ---------
# [user_id_num =30000,creative_id_num = 11500, epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.3903043457349141，精确度 = 0.8678666949272156
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test)= 82.2%
# ---------
# [user_id_num =30000,creative_id_num = 15000, epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.38081181917190554，精确度 = 0.8708000183105469
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  80.34825870646766 %
# ---------
# [user_id_num =30000,creative_id_num = 17000, epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.38891106843948364，精确度 = 0.8706666827201843
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  80.43117744610282 %
# ---------
# [user_id_num =30000,creative_id_num = 20000, epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.3896093798955282，精确度 = 0.8717333078384399
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  79.76782752902156 %
# ---------
# [user_id_num =35000,creative_id_num = int(user_id_num * 2 / 3), epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.4136177837235587，精确度 = 0.8762285709381104
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  76.69971671388102 %
# ---------
# [user_id_num =37350,creative_id_num = int(user_id_num * 2 / 3), epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.3889377001432822，精确度 = 0.8657100200653076
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  82.71767810026385 %
# ---------
# [user_id_num =40000,creative_id_num = int(user_id_num * 2 / 3), epochs = 7, embedding_size=creative_id_num/100]
# 模型预测-->损失值 = 0.37648941814899445，精确度 = 0.871999979019165
# sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  78.43137254901961 %
# 用户和素材数目增长到一定程度，就不会再增长，是因为新的数据，噪声增加的部分已经超过信息增加的部分
user_id_num = 28000
creative_id_num = int(user_id_num * 2 / 3)
batch_size = int(user_id_num / 30)
embedding_size = int(creative_id_num / 100)
epochs = 7
# click_log 中除了一个用户访问了 16868 个素材，其他都在 1706 以内
# number_user_id_1m 中除了一个用户访问了924个素材，其他都在303以内
max_len = 304

X_doc = np.array(np.zeros([user_id_num], dtype = bool), dtype = object)
y_doc = np.zeros([user_id_num])
for i, row_data in enumerate(X_data):
    user_id = row_data[2] - 1
    if user_id < user_id_num:
        if X_doc[user_id] is False:
            X_doc[user_id] = []
            pass
        creative_id = row_data[1]
        if creative_id >= creative_id_num:
            creative_id = 1
            pass
        X_doc[user_id].append(creative_id)
        y_doc[user_id] = y_data[i]
        pass
    pass
# padding: 字符串，'pre' 或 'post' ，在序列的前端补齐还是在后端补齐。
X_padded = pad_sequences(X_doc, maxlen = max_len, padding = 'post')
y_padded = y_doc
X = X_padded
y = y_padded

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, stratify = y)
print("\t训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))

X_train_scaled = X_train
y_train_scaled = y_train
X_test_scaled = X_test
y_test_scaled = y_test

print("* 构建网络")
model = Sequential()
model.add(Embedding(creative_id_num, embedding_size, input_length = max_len))
model.add(Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(
        64, activation = keras.activations.relu,
        kernel_regularizer = keras.regularizers.l1(0.00001)
))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(
        32, activation = keras.activations.relu,
        kernel_regularizer = keras.regularizers.l1(0.00001)
))
model.add(Dense(1, activation = 'sigmoid'))
# compile the model
print("* 编译模型")
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
print(model.summary())

print("* 验证模型→留出验证集")
split_number = int(user_id_num * 0.1)
X_val_scaled = X_train_scaled[:split_number]
y_val_scaled = y_train_scaled[:split_number]
partial_X_train_scaled = X_train_scaled[split_number:]
partial_y_train_scaled = y_train_scaled[split_number:]

print("\tX_val.shape =", X_val_scaled.shape)
print("\tpartial_X_train.shape =", partial_X_train_scaled.shape)
print("\ty_val.shape =", y_val_scaled.shape)
print("\tpartial_y_train.shape =", partial_y_train_scaled.shape)

print("训练数据 =", X_val_scaled[0], y_val_scaled[0])
print("* 训练模型")
# 模型越复杂，精度有时可以提高，但是过拟合出现的时间就越早 (epochs 发生过拟合的值越小)
# 过拟合了，训练集的精度越来越高，测试集的精度开始下降
# verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
model.fit(partial_X_train_scaled, partial_y_train_scaled, epochs = epochs, batch_size = batch_size,
          validation_data = (X_val_scaled, y_val_scaled), use_multiprocessing = True, verbose = 2)

# # 训练全部数据，不分离出验证数据，TODO:效果不好？
# model.fit(X_train_scaled, y_train_scaled, epochs = 6, batch_size = batch_size,
#           use_multiprocessing = True, verbose = 2)
results = model.evaluate(X_test_scaled, y_test, verbose = 0)
predictions = model.predict(X_test_scaled).squeeze()
print("模型预测-->", end = '')
print("损失值 = {}，精确度 = {}".format(results[0], results[1]))
print("sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) = ",
      sum(abs(np.array(predictions > 0.5, dtype = int) - y_test_scaled)) / sum(y_test_scaled) * 100,
      '%')
print("前10个真实的目标数据 =", y_test[:10])
print("前10个预测的目标数据 =", np.array(predictions[:10] > 0.5, dtype = int))
print("sum(predictions>0.5) =", sum(predictions > 0.5))
print("sum(y_test_scaled) =", sum(y_test_scaled))
print("sum(abs(predictions-y_test_scaled))=",
      sum(abs(np.array(predictions > 0.5, dtype = int) - y_test_scaled)))
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(900, 1000)
    winsound.Beep(600, 500)
    winsound.Beep(900, 1000)
    winsound.Beep(600, 500)
plt.show()

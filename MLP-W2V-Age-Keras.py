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
# 索引在数据库中是从 1 开始的，加上 2 ，是因为 Python 的索引是从 0 开始的
# 并且需要保留 {0, 1, 2} 三个数：0 表示 “padding”（填充），1 表示 “start”（用户开始），2 表示 “unknown”（未知词）
X_data[:, 1] = X_data[:, 1] + 2

user_id_num = 28000
creative_id_num = int(user_id_num * 2 / 3)
batch_size = int(user_id_num / 30)
embedding_size = int(creative_id_num / 100)
epochs = 14

# click_log 中除了一个用户访问了 16868 个素材，其他都在 1706 以内
# all_log_valid_1m 中除了一个用户访问了1032次素材，其他都在343次以内
max_len = 352  # 352 可以被 16 整除

X_doc = np.zeros([user_id_num], dtype = object)
y_doc = np.zeros([user_id_num])
# X_novalid_doc = np.array(np.zeros([373489 - user_id_num], dtype = bool), dtype = object)
# y_novalid_doc = np.zeros([373489 - user_id_num])
tmp_user_id = -1  # -1 表示 id 不在数据序列中
tmp_time_id = 0
# 生成的用户序列数据：1 表示用户访问序列的开始；0 表示这天没有访问素材；2 表示这个素材不在词典中
# 序列中重复的数据是因为某个素材访问好几次；最后的0是填充数据
for i, row_data in enumerate(X_data):
    time_id = row_data[0]
    creative_id = row_data[1]
    user_id = row_data[2] - 1  # 索引从 0 开始
    click_times = row_data[3]

    # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
    if user_id < user_id_num:
        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
        if user_id != tmp_user_id:
            tmp_user_id = user_id
            tmp_time_id = 0
            X_doc[user_id] = [1]  # 1 表示序列的开始
            # 新建用户序列时，更新用户的标签
            y_doc[user_id] = y_data[i]
            pass
        if tmp_time_id < time_id:
            # 按照时间差更新用户序列中的空缺天数，两个天数之间的空缺天数=后-次的天数-前一次的天数-1
            X_doc[user_id].extend([0 for _ in range(time_id - tmp_time_id - 1)])
            tmp_time_id = time_id
            pass
        # 超过词典大小的素材标注为 2，即「未知」
        if creative_id >= creative_id_num:
            creative_id = 2
            pass
        X_doc[user_id].extend([creative_id for _ in range(click_times)])  # 按照点击次数更新用户序列
        pass
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

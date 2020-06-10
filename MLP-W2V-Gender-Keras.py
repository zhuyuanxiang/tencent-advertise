# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   MLP-W2V-Gender-Keras.py
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
print("数据加载完成。")

# ----------------------------------------------------------------------
print("* 清洗数据集")
user_id_num = 50000  # 用户数
creative_id_num = 50000  # 素材数
max_len = 32

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from preprocessing import (data_sequence, data_sequence_times, data_sequence_times_with_interval)

# data_sequence()
#       点击次数超过 1 次的也只有一条数据，没有数据的天就跳过(不填充0)
# data_sequence_times()
#       点击次数超过 1 次的数据重复生成，没有数据的天就跳过(不填充0)
# data_sequence_times_with_interval()
#       点击次数超过 1 次的数据重复生成，没有数据的天数就插入一个0，不管差几天
# data_sequence_times_with_empty()
#       点击次数超过 1 次的数据重复生成，没有数据的天就置 0，差几天就填充几个 0
# data_sequence_with_fix()
#       每天固定几个数据，没有数据的天就置0，超过范围的数据就截断，点击次数超过 1 次的也只有一条数据，

X_doc, y_doc = data_sequence_times_with_interval(X_data, y_data, user_id_num, creative_id_num)
# padding: 字符串，'pre' 或 'post' ，在序列的前端补齐还是在后端补齐。
X = pad_sequences(X_doc, maxlen = max_len, padding = 'post')
y = y_doc

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, stratify = y)
print("\t训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))

# ----------------------------------------------------------------------
from network import (construct_Bidirectional_LSTM, construct_Conv1d, construct_Conv1d_LSTM,
                     construct_LSTM, construct_mlp, )

# embedding_size = int(creative_id_num / 100)
embedding_size = 64
# model=construct_mlp(creative_id_num, embedding_size, max_len)
model = construct_Conv1d_LSTM(creative_id_num, embedding_size, max_len)
# model = construct_Conv1d(creative_id_num, embedding_size, max_len)
# compile the model
print("* 编译模型")
model.compile(optimizer = optimizers.RMSprop(lr = 2e-04),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
print(model.summary())

# ----------------------------------------------------------------------
print("* 训练模型")
print("训练数据 =", X_train[0], y_train[0])

# 模型越复杂，精度有时可以提高，但是过拟合出现的时间就越早 (epochs 发生过拟合的值越小)
# 过拟合了，训练集的精度越来越高，测试集的精度开始下降
# verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
epochs = 20
# batch_size = int(user_id_num / 30)
batch_size = 256
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
          validation_split = 0.2, use_multiprocessing = True, verbose = 2)

# # 训练全部数据，不分离出验证数据，TODO:效果不好？
# model.fit(X_train_scaled, y_train_scaled, epochs = 6, batch_size = batch_size,
#           use_multiprocessing = True, verbose = 2)

# ----------------------------------------------------------------------
results = model.evaluate(X_test, y_test, verbose = 0)
predictions = model.predict(X_test).squeeze()
print("模型预测-->", end = '')
print("损失值 = {}，精确度 = {}".format(results[0], results[1]))
print("sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) = ",
      sum(abs(np.array(predictions > 0.5, dtype = int) - y_test)) / sum(y_test) * 100,
      '%')
print("前10个真实的目标数据 =", y_test[:10])
print("前10个预测的目标数据 =", np.array(predictions[:10] > 0.5, dtype = int))
print("sum(predictions>0.5) =", sum(predictions > 0.5))
print("sum(y_test_scaled) =", sum(y_test))
print("sum(abs(predictions-y_test_scaled))=",
      sum(abs(np.array(predictions > 0.5, dtype = int) - y_test)))
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(900, 1000)
    winsound.Beep(600, 500)
    winsound.Beep(900, 1000)
    winsound.Beep(600, 500)
plt.show()

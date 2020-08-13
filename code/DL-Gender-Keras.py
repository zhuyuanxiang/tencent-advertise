# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   Train-Single-Input-Model.py
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
# 加载数据
from preprocessing import data_sequence, load_data, data_sequence_no_start

file_name = './data/train_data.csv'
X_data, y_data = load_data(file_name,label_name='age')

# ----------------------------------------------------------------------
# 定义全局变量
user_id_max = 50000  # 用户数
creative_id_end = 50000  # 素材数
max_len = 16

# ----------------------------------------------------------------------
# 清洗数据集
# X_doc, y_doc = data_sequence(X_data, y_data, user_id_max, creative_id_num)
X_doc, y_doc = data_sequence_no_start(X_data, y_data, user_id_max, creative_id_end)
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
from network import construct_GlobalMaxPooling1D

embedding_size = 128
model = construct_GlobalMaxPooling1D(creative_id_end, embedding_size, max_len)
# ----------------------------------------------------------------------
print("* 编译模型")
RMSProp_lr = 6e-04
model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
print(model.summary())

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
print("sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) = error% =",
      sum(abs(np.array(predictions > 0.5, dtype = int) - y_test)) / sum(y_test) * 100,
      '%')
print("前100个真实的目标数据 =", np.array(y_test[:100], dtype = int))
print("前100个预测的目标数据 =", np.array(predictions[:100] > 0.5, dtype = int))
print("sum(predictions>0.5) =", sum(predictions > 0.5))
print("sum(y_test) =", sum(y_test))
print("sum(abs(predictions-y_test))=error_number=",
      sum(abs(np.array(predictions > 0.5, dtype = int) - y_test)))
print("实验报告参数")
print("user_id_maxber =", user_id_max)
print("creative_id_number =", creative_id_end)
print("max_len =", max_len)
print("embedding_size =", embedding_size)
print("epochs =", epochs)
print("batch_size =", batch_size)
print("RMSProp =", RMSProp_lr)

# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1500)
    plt.show()

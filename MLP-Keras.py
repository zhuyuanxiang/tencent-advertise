# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   tencent_mlp.py
@Version    :   v0.1
@Time       :   2020-05-26 17:07
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :   使用 MLP 处理 腾迅广告大赛的数据
"""
import os
import sys
import sklearn
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd
import winsound
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

print("* 加载数据集...")

# 「CSV」文件字段名称
# "time_id","user_id_inc","user_id","creative_id_inc","creative_id","click_times","age","gender"
filename = './data/train_data.csv'
df = pd.read_csv(filename)

# y_data = (df['age'].values + df['gender'] * 10).values  # 年龄和性别一起作为目标数据
y_data = df['age'].values - 1  # 年龄作为目标数据
# y_data = df['gender'].values - 1  # 性别作为目标数据
# 选择需要的列作为输入数据
X_data = df[["time_id", "creative_id_inc", "user_id_inc", "click_times"]].values

# [age]
# [11000,5500]
# 模型预测-->损失值 = 2.0421775305487895，精确度 = 0.22363635897636414
# [12000,6000]
# 模型预测-->损失值 = 2.015912573814392，精确度 = 0.22066666185855865
# [gender]
# [11000,5500]
# 模型预测-->损失值 = 0.39888836656917226，精确度 = 0.8665454387664795
# sum(predictions>0.5) = 257
# sum(y_test) = 454.0
# sum(abs(predictions-y_test))= 516.1640199696994
# [12000,6000]
# 模型预测-->损失值 = 0.38617609429359434，精确度 = 0.8693333268165588
# sum(predictions>0.5) = 275
# sum(y_test) = 487.0
# sum(abs(predictions-y_test))= 552.8971114754677
# [13000,6500]
# 模型预测-->损失值 = 0.40163239833024833，精确度 = 0.8624615669250488
# sum(predictions>0.5) = 291
# sum(y_test) = 530.0
# sum(abs(predictions-y_test))= 598.6849186713807
# [14000,7000]
# 模型预测-->损失值 = 0.4230275722231184，精确度 = 0.8557142615318298
# sum(predictions>0.5) = 384
# sum(y_test) = 573.0
# sum(abs(predictions-y_test))= 635.9497097495478
user_id_num = 12000
creative_id_end = 6000
data_shape = (creative_id_end,)
batch_size = int(user_id_num / 30)
epochs = int(user_id_num / 150)

# 对数据进行 One-Hot 编码，但是不是只有一个「1」的编码
# 以每个用户91天访问了哪些素材进行编码，维度是素材库的大小，访问的素材为1，没有访问的素材为0，
# {素材(creative_id)库的大小(维度)：20000、用户(user_id_inc)库的大小(维度)：30000}
# creative_id_inc: 表示素材的编码，这个编码已经处理过，按照所有素材的 tf-idf 值逆序排列(大的在前面)
# 超过维度的素材编码为0，表示未知素材
# user_id_inc：表示用户的编码，这个编码已经处理过，按照 91天 内访问素材的数量逆序排列(大的在前面)
# 超过维度的用户放弃
# time_id：表示91天内第几天访问的广告
X_one_hot = np.zeros([user_id_num, creative_id_end], dtype = np.bool)
y_one_hot = np.zeros([user_id_num])
for i, row_data in enumerate(X_data):
    user_id = row_data[2] - 1
    if user_id < user_id_num:
        creative_id = row_data[1]
        if creative_id >= creative_id_end:
            creative_id = 0
            pass
        X_one_hot[user_id, creative_id] = True
        y_one_hot[user_id] = y_data[i]
        pass
    pass

X = np.array(X_one_hot, dtype = np.int8)
y = y_one_hot

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, stratify = y)
print("\t训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))

X_train_scaled = X_train
X_test_scaled = X_test

print("* 构建 MLP 网络")


def creative_age_model():
    age_model = keras.Sequential()
    # age_model.add(keras.layers.Dropout(0.2))
    # age_model.add(keras.layers.Dense(128, activation = keras.activations.relu))
    # age_model.add(keras.layers.Dropout(0.2))
    # age_model.add(keras.layers.Dense(128, activation = keras.activations.relu))
    # age_model.add(keras.layers.Dropout(0.5))
    age_model.add(
            keras.layers.Dense(64, activation = keras.activations.relu, input_shape = data_shape))
    age_model.add(keras.layers.Dense(10, activation = keras.activations.softmax))
    return age_model


def creative_gender_model():
    embedding_size=200
    gender_model = keras.Sequential()
    gender_model.add(keras.layers.Dense(
            64, activation = keras.activations.relu,
            input_shape = data_shape,
            kernel_regularizer = keras.regularizers.l1(0.00001)
    ))
    # gender_model.add(keras.layers.Dense(
    #         64, activation = keras.activations.relu,
    #         kernel_regularizer = keras.regularizers.l1(0.00001)
    # ))
    gender_model.add(keras.layers.Dense(
            32, activation = keras.activations.relu,
            kernel_regularizer = keras.regularizers.l1(0.00001)
    ))
    gender_model.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))
    return gender_model


model=creative_age_model()
# model = creative_gender_model()
model.summary()

print("* 编译模型")
# 对「gender」字段进行学习
# model.compile(optimizer = keras.optimizers.RMSprop(lr = 0.0001),
#               # loss = keras.losses.mean_squared_error,
#               # metrics = [keras.metrics.mean_squared_error])
#               loss = keras.losses.binary_crossentropy,
#               metrics = [keras.metrics.binary_accuracy])
# # 对「age」字段进行学习
model.compile(optimizer = keras.optimizers.RMSprop(lr = 0.0001),
              loss = keras.losses.sparse_categorical_crossentropy,
              metrics = [keras.metrics.sparse_categorical_accuracy])

print("* 验证模型→留出验证集")
split_number = int(user_id_num * 0.1)
X_val_scaled = X_train_scaled[:split_number]
partial_X_train_scaled = X_train_scaled[split_number:]

# y_val_one_hot = y_train_one_hot[:split_number]
# partial_y_train_one_hot = y_train_one_hot[split_number:]
y_val = y_train[:split_number]
partial_y_train = y_train[split_number:]

print("\tX_val.shape =", X_val_scaled.shape)
print("\tpartial_X_train.shape =", partial_X_train_scaled.shape)
print("\ty_val.shape =", y_val.shape)
print("\tpartial_y_train.shape =", partial_y_train.shape)

print("* 训练模型")
# 模型越复杂，精度有时可以提高，但是过拟合出现的时间就越早 (epochs 发生过拟合的值越小)
# 过拟合了，训练集的精度越来越高，测试集的精度开始下降
# verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
model.fit(partial_X_train_scaled, partial_y_train, epochs = epochs, batch_size = batch_size,
          validation_data = (X_val_scaled, y_val), use_multiprocessing = True, verbose = 2)

# model.fit(X_train_scaled, y_train, epochs = 60, batch_size = batch_size,
#           use_multiprocessing = True,verbose = 2)
results = model.evaluate(X_test_scaled, y_test, verbose = 0)
predictions = model.predict(X_test_scaled).squeeze()
print("模型预测-->", end = '')
print("损失值 = {}，精确度 = {}".format(results[0], results[1]))
# print("sum(predictions>0.5) =", sum(predictions > 0.5))
# print("sum(y_test) =", sum(y_test))
# print("sum(abs(predictions-y_test))=", sum(abs(predictions - y_test)))
print("前10个真实的目标数据 =", y_test[:10])
print("模型预测，前10个预测结果 =", np.argmax(predictions[:10],1))

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
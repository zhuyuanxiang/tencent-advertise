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
from tensorflow import keras  # keras 也可以使用高版本的TensorFlow自带的

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
assert np.__version__ in ["1.16.5", "1.16.4"]

print("* 加载数据集...")

# 「CSV」文件字段名称
# "creative_id","click_times","ad_id","product_id","product_category","advertiser_id","industry",
filename = './data/all_log_300k.csv'
df = pd.read_csv(filename)

# y_data = (df['age'].values + df['gender'] * 10).values  # 年龄和性别一起作为目标数据
# y_data = df['age'].values - 1  # 年龄作为目标数据
y_data = df['gender'].values - 1  # 性别作为目标数据
# 选择需要的列作为输入数据
X_data = df[[
        "creative_id",
        "click_times",
        "ad_id",
        "product_id",
        "product_category",
        "advertiser_id",
        "industry",
]].values
data_shape = (7,)
X = X_data
y = y_data
max_len = len(y_data)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed,stratify = y)
train_len = len(y_train)
test_len = len(y_test)
print("\t训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % (train_len, test_len))

# # 编码目标数据，使用 sklearn 的 One-Hot 编码
# 取消 One-Hot 编码，因为使用其他损失函数即可

print("* 归一化输入数据...")
# StandardScaler    ：确保每个特征的平均值为0，方差为1，使所有特征都位于同一量级。
# RobustScaler      ：确保每个特征的统计属于都位于同一范围，中位数为0，四分位数为1？，从而忽略异常值
# MinMaxScaler      ：确保所有特征都位于0到1之间
# Normalizer        ：归一化。对每个数据点都进行缩放，使得特征向量的欧氏长度为1。
#                       即将数据点都投向到半径为1的圆上。
#                       因此，只关注数据的方向（或角度），不关注数据的长度。


from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled = X_train
# X_test_scaled = X_test

print("* 构建 MLP 网络")
model = keras.Sequential()
model.add(keras.layers.Dense(128, activation = keras.activations.relu, input_shape = data_shape))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(128, activation = keras.activations.relu))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(128, activation = keras.activations.relu))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation = keras.activations.relu))
# model.add(keras.layers.Dense(64, activation = keras.activations.relu, input_shape = (7,)))
model.add(keras.layers.Dense(32, activation = keras.activations.relu))
model.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))
# model.add(keras.layers.Dense(10, activation = keras.activations.softmax))
model.summary()

print("* 编译模型")
# 对「age」字段进行学习
model.compile(optimizer = keras.optimizers.RMSprop(lr = 0.0001),
              loss = keras.losses.binary_crossentropy,
              metrics = [keras.metrics.binary_accuracy])
# # 对「gender」字段进行学习
# model.compile(optimizer = keras.optimizers.RMSprop(lr = 0.0001),
#               loss = keras.losses.sparse_categorical_crossentropy,
#               metrics = [keras.metrics.sparse_categorical_accuracy])

print("* 验证模型→留出验证集")
split_number = 20000
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
# # 模型越复杂，精度有时可以提高，但是过拟合出现的时间就越早
# # epochs = 9 就出现过拟合了，训练集的精度越来越高，测试集的精度开始下降
# # verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
model.fit(partial_X_train_scaled, partial_y_train, epochs = 50, batch_size = 1024,
          validation_data = (X_val_scaled, y_val), use_multiprocessing = True, verbose = 2)
print("\t模型预测-->", end = '')
results = model.evaluate(X_test_scaled, y_test, verbose = 0)
print("\t损失值 = {}，精确度 = {}".format(results[0], results[1]))
predictions = model.predict(X_train_scaled)
print("\t前10个真实的目标数据 =", y_test[:10])
print("\t前10个预测结果中最大值所在列 =", end = '')
print(np.argmax(predictions[:10], 1))
print("\t模型预测，前10个预测结果 =")
print(predictions[:10])
#
# # history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512,
# #                     validation_data = (x_val, y_val), use_multiprocessing = True, verbose = 2)
# # print("\t模型预测-->", end = '')
# # results = model.evaluate(x_test, one_hot_test_labels, verbose = 0)
# # print("\t损失值 = {}，精确度 = {}".format(results[0], results[1]))
# # predictions = model.predict(x_test)
# # print("\t前10个真实的目标数据 =",test_labels[:10])
# # print("\t前10个预测结果中最大值所在列 =",end = '')
# # print(np.argmax(predictions[:10], 1))
# # # print("\t前10个预测结果（每条数据有46个分类概率，最大概率为最可能的分类） =")
# # # print(predictions[:10])
# #
# # history_dict = history.history
# # # print("\thistory_dict.keys() =", history_dict.keys())
# # loss_values = history_dict['loss']
# # val_loss_values = history_dict['val_loss']
# # epochs_range = range(1, len(loss_values) + 1)
# #
# # plt.figure()
# # plt.plot(epochs_range, loss_values, 'bo', label = '训练集的损失')  # bo 蓝色圆点
# # plt.plot(epochs_range, val_loss_values, 'b', label = '验证集的损失')  # b 蓝色实线
# # plt.title('图3-7：训练损失和验证损失')
# # plt.xlabel('Epochs--批次')
# # plt.ylabel('Loss--损失')
# # plt.legend()
# #
# # acc = history_dict['categorical_accuracy']
# # val_acc = history_dict['val_categorical_accuracy']
# #
# # plt.figure()
# # plt.plot(epochs_range, acc, 'bo', label = '训练集的精度')
# # plt.plot(epochs_range, val_acc, 'b', label = '验证集的精度')
# # plt.title('图3-8：训练精度和验证精度')
# # plt.xlabel('Epochs--批次')
# # plt.ylabel('Accuracy--精度')
# # plt.ylim([0., 1.2])
# # plt.legend()
# #
# # # 标签采用整数编码与采用 One-Hot 编码的区别
# # # 本质是相同的
# # # 损失函数由 categorical_crossentropy 换成 sparse_categorical_crossentropy
# # # y_train = np.array(train_labels)
# # # y_test = np.array(test_labels)
# # # model.compile(optimizer = keras.optimizers.rmsprop(lr = 0.001),
# # #               loss=keras.losses.sparse_categorical_crossentropy,
# # #               metrics = [keras.metrics.binary_accuracy])
# # # 运行结束的提醒
# # winsound.Beep(600, 500)
# # if len(plt.get_fignums()) != 0:
# #     plt.show()
# # pass

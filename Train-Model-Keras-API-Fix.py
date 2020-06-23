# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   Train-Model-Keras-API.py
@Version    :   v0.1
@Time       :   2020-06-15 8:36
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# common imports
import os
import pickle
import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import winsound
from tensorflow.python.keras import backend as K
from tensorflow import keras

from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras import losses, metrics, optimizers
from tensorflow.python.keras.layers import Activation, BatchNormalization, concatenate, Dropout, Bidirectional, Conv1D, \
    MaxPooling1D, Flatten, GRU
from tensorflow.python.keras.layers import Dense, Embedding, GlobalMaxPooling1D, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.regularizers import l2

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


# ==================================================
# 从文件中载入数据
def load_data():
    # 「CSV」文件字段名称
    df = pd.read_csv(file_name, dtype = int)
    # --------------------------------------------------
    # 输入数据处理：选择需要的列
    X_csv = df[field_list].values
    # 索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的，因此字段的偏移量为 -1
    # 没有在数据库中处理索引，是因为尽量不在数据库中修正原始数据，除非是不得不变更的数据，这样子业务逻辑清楚
    # user_id_inc:      X_csv[:,0]
    # creative_id_inc:  X_csv[:,1]
    # time_id:          X_csv[:,2]
    # click_times:      X_csv[:,3] 数据是值，不需要减 1
    for j in range(field_num - 1):
        X_csv[:, j] = X_csv[:, j] - 1
        pass
    # 生成 creative_id: 0 表示 “padding”（填充），1 表示 “unknown”（未知词）
    # 'creative_id_inc' 字段的偏移量为 2，是因为需要保留 {0, 1}
    X_csv[:, 1] = X_csv[:, 1] + 2
    # --------------------------------------------------
    # 目标数据处理：目标字段的偏移量是 -1，是因为索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的
    # 既可以加载 'age'，也可以加载 'gender'
    y_csv = df[label_name].values - 1
    print("数据加载完成。")
    return X_csv, y_csv


# ==================================================
# 生成所需要的数据
def generate_fix_data(X_data, y_data):
    print("数据生成中（共 {0} 条数据)：".format(30000000), end = '')
    y_doc = np.zeros([user_id_num], dtype = int)
    # 初始化 X_doc 为空的列表
    # X_doc[:,0]: creative_id
    # X_doc[:,1]: click_times
    train_field_num = field_num - 2  # filed_list 去除 user_id, time_id
    X_doc = np.zeros([user_id_num, train_field_num, time_id_max * period_length // period_days], dtype = object)
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    prev_user_id = -1
    prev_time_id = 0
    period_index = 0
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print("第 {0} 条数据-->".format(i), end = ';')
            pass
        user_id = row_data[0]
        time_id = row_data[2]
        if user_id >= user_id_num:
            break
        y_doc[user_id] = y_data[i] if age_sigmoid == -1 or label_name == 'gender' else int(age_sigmoid == y_data[i])
        # 整理过的数据已经按照 user_id 的顺序编号，当 user_id 变化时，就代表前一个用户的数据已经清洗完成
        if user_id > prev_user_id:
            # 重置临时变量
            prev_user_id = user_id
            prev_time_id = 0
            period_index = 0
            pass

        # tmp_time_id = time_id - prev_time_id
        # if tmp_time_id > 0:  # 如果日期发生变化就需要重置 prev_time_id
        #     prev_time_id = time_id
        #     if tmp_time_id >= 7 or week_length == period_index:  # 如果两个日期差距超过7天就需要重置 period_index
        #         period_index = 0

        if time_id - prev_time_id >= period_days:
            prev_time_id = time_id // period_days * period_days
            period_index = 0

        if period_index == period_length:  # 每周访问素材填满后不再填充
            continue

        # row_data[0]: user_id
        creative_id = row_data[1]  # 这个值已经在读取时修正过，增加了偏移量 2，保留了 {0,1}
        # row_data[2]: time_id
        click_times = row_data[3]  # 这个不是词典，本身就是值，不需要再调整

        # 素材是关键
        if creative_id_end > creative_id > creative_id_begin:
            creative_id = creative_id - creative_id_begin
        elif creative_id < creative_id_end - creative_id_max:
            creative_id = creative_id_max - creative_id_begin + creative_id
        else:
            creative_id = 1  # 超过词典大小的素材标注为 1，即「未知」

        X_doc[user_id, 0, (time_id // period_days) * period_length + period_index] = creative_id
        X_doc[user_id, 1, (time_id // period_days) * period_length + period_index] = click_times
        period_index = period_index + 1
        pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ==================================================
def output_example_data(X, y):
    print("数据(X[0], y[0]) =", X[0], y[0])
    print("数据(X[30], y[30]) =", X[30], y[30])
    print("数据(X[600], y[600]) =", X[600], y[600])
    if len(y) > 8999:
        print("数据(X[8999], y[8999]) =", X[8999], y[8999])
    if len(y) > 119999:
        print("数据(X[119999], y[119999]) =", X[119999], y[119999])
    if len(y) > 224999:
        print("数据(X[224999], y[224999]) =", X[224999], y[224999])
    if len(y) > 674999:
        print("数据(X[674999], y[674999]) =", X[674999], y[674999])
    if len(y) > 899999:
        print("数据(X[899999], y[899999]) =", X[899999], y[899999])
        pass
    pass


# ==================================================
# 损失函数建议不再变更
def myCrossEntropy(y_true, y_pred, e = 0.3):
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss0 = K.sparse_categorical_crossentropy(K.zeros_like(y_true), y_pred)
    loss1 = K.sparse_categorical_crossentropy(K.ones_like(y_true), y_pred)
    loss2 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 2, y_pred)
    loss3 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 3, y_pred)
    loss4 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 4, y_pred)
    loss5 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 5, y_pred)
    loss6 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 6, y_pred)
    loss7 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 7, y_pred)
    loss8 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 8, y_pred)
    loss9 = K.sparse_categorical_crossentropy(K.ones_like(y_true) * 9, y_pred)
    return (
            (100.0 - 5.765 - 1.359 - 1.000 - 1.348 - 1.554 - 1.995 - 3.042 - 6.347 - 10.431 - 17.632) * loss
            + 5.765 * loss0 + 1.359 * loss1 + 1.000 * loss2 + 1.348 * loss3 + 1.553 * loss4
            + 1.995 * loss5 + 3.042 * loss6 + 6.347 * loss7 + 10.421 * loss8 + 17.632 * loss9
    )
    # return loss + loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9


# 训练网络模型
def construct_model():
    input_creative_id = Input(
        shape = (time_id_max * period_length // period_days), dtype = 'int32', name = 'creative_id'
    )
    embedded_creative_id = Embedding(
        creative_id_window, embedding_size, name = 'creative_id_embedded')(input_creative_id)
    # encoded_creative_id = GlobalMaxPooling1D(name = 'creative_id_encoded')(embedded_creative_id)
    x = Conv1D(embedding_size, 3, 2, activation = 'relu', name = 'creative_id_convolution_0101')(embedded_creative_id)
    x = Dropout(0.5, name = 'creative_id_D_0101')(x)
    x = BatchNormalization(name = 'creative_id_BN_0101')(x)
    x = MaxPooling1D(8, name = 'creative_id_pool_0101')(x)
    x = Dropout(0.5, name = 'creative_id_D_0102')(x)
    x = BatchNormalization(name = 'creative_id_BN_0102')(x)
    # x = Conv1D(embedding_size, 3, 2, activation = 'relu', name = 'creative_id_convolution_0102')(x)
    # x = Dropout(0.5, name = 'creative_id_D_0201')(x)
    # x = BatchNormalization(name = 'creative_id_BN_0201')(x)
    # # x = MaxPooling1D(3, name = 'creative_id_pool_0102')(x)
    # # x = Dropout(0.5, name = 'creative_id_D_0202')(x)
    # # x = BatchNormalization(name = 'creative_id_BN_0202')(x)
    x = GRU(embedding_size, dropout = 0.5, recurrent_dropout = 0.5,
            return_sequences = True, name = 'creative_id_GRU_0101')(x)
    encoded_creative_id = GlobalMaxPooling1D(name = 'creative_id_encoded')(x)

    # # LSTM(14) : 是因为 91 天正好是 14 个星期
    # # LSTM(32) : 方便计算
    # input_click_times = Input(
    #     shape = (time_id_max * period_length // period_days, 1), dtype = 'float32', name = 'click_times'
    # )
    # x = Conv1D(1, 3, 2, activation = 'relu', name = 'click_times_convolution_0101')(input_click_times)
    # x = Dropout(0.5, name = 'click_times_D_0101')(x)
    # x = BatchNormalization(name = 'click_times_BN_0101')(x)
    # x = MaxPooling1D(8, name = 'click_times_pool_0101')(x)
    # x = Dropout(0.5, name = 'click_times_D_0102')(x)
    # x = BatchNormalization(name = 'click_times_BN_0102')(x)
    # x = Conv1D(embedding_size, 2, 1, activation = 'relu', name = 'click_times_convolution_0102')(x)
    # x = Dropout(0.5, name = 'click_times_D_0201')(x)
    # x = BatchNormalization(name = 'click_times_BN_0201')(x)
    # # x = MaxPooling1D(3, name = 'click_times_pool_0102')(x)
    # # x = Dropout(0.5, name = 'click_times_D_0202')(x)
    # # x = BatchNormalization(name = 'click_times_BN_0202')(x)
    # x = GRU(1, dropout = 0.5, recurrent_dropout = 0.5,
    #         return_sequences = True, name = 'click_times_GRU_0101')(x)
    # encoded_click_times = Flatten(name = 'click_times_encoded')(x)
    #
    # concatenated = concatenate([
    #     encoded_creative_id,
    #     encoded_click_times,
    #     # encoded_product_id,
    #     # encoded_category,
    #     # encoded_advertiser_id,
    #     # encoded_industry
    # ], axis = -1)
    #
    # x = Dropout(0.5)(concatenated)
    x = Dropout(0.5, name = 'Dense_Dropout_0101')(encoded_creative_id)
    x = Dense((embedding_size + 0), kernel_regularizer = l2(0.001), name = 'Dense_0101')(x)
    x = BatchNormalization(name = 'Dense_BN_0101')(x)
    x = Activation('relu', name = 'Dense_Activation_0101')(x)

    x = Dropout(0.5, name = 'Dense_Dropout_0102')(x)
    x = Dense((embedding_size + 0), kernel_regularizer = l2(0.001), name = 'Dense_0102')(x)
    x = BatchNormalization(name = 'Dense_BN_0102')(x)
    x = Activation('relu', name = 'Dense_Activation_0102')(x)

    x = Dropout(0.5, name = 'Dense_Dropout_0201')(x)
    x = Dense((embedding_size + 0) // 2, kernel_regularizer = l2(0.001), name = 'Dense_0201')(x)
    x = BatchNormalization(name = 'Dense_BN_0201')(x)
    x = Activation('relu', name = 'Dense_Activation_0201')(x)

    x = Dropout(0.5, name = 'Dense_Dropout_0202')(x)
    x = Dense((embedding_size + 0) // 2, kernel_regularizer = l2(0.001), name = 'Dense_0202')(x)
    x = BatchNormalization(name = 'Dense_BN_0202')(x)
    x = Activation('relu', name = 'Dense_Activation_0202')(x)

    if label_name == "age" and age_sigmoid == -1:
        x = Dropout(0.5)(x)
        x = Dense(10, kernel_regularizer = l2(0.001), name = 'output')(x)
        x = BatchNormalization()(x)
        output_tensor = Activation('softmax')(x)

        model = Model([
            input_creative_id,
            # input_click_times,
            # input_product_id,
            # input_category,
            # input_advertiser_id,
            # input_industry
        ], output_tensor)

        print('-' * 5 + ' ' * 3 + "编译模型" + ' ' * 3 + '-' * 5)

        model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
                      loss = myCrossEntropy,
                      # loss = losses.sparse_categorical_crossentropy,
                      metrics = [metrics.sparse_categorical_accuracy])
    elif label_name == 'gender' or age_sigmoid != -1:
        x = Dropout(0.5)(x)
        x = Dense(1, kernel_regularizer = l2(0.001), name = 'output')(x)
        x = BatchNormalization()(x)
        output_tensor = Activation('sigmoid')(x)

        model = Model([
            input_creative_id,
            # input_click_times,
            # input_product_id,
            # input_category,
            # input_advertiser_id,
            # input_industry
        ], output_tensor)

        print('-' * 5 + ' ' * 3 + "编译模型" + ' ' * 3 + '-' * 5)

        model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
                      loss = losses.binary_crossentropy,
                      metrics = [metrics.binary_accuracy])
    else:
        raise Exception("错误的标签类型！")
    return model


# ==================================================
def output_parameters():
    print("实验报告参数")
    print("\t label_name =", label_name)
    print("\t user_id_number =", user_id_num)
    print("\t time_id_max =", time_id_max)
    print("\t creative_id_begin =", creative_id_begin)
    print("\t creative_id_end =", creative_id_end)
    print("\t RMSProp =", RMSProp_lr)
    print("\t batch_size =", batch_size)
    print("\t embedding_size =", embedding_size)
    print("\t day_length =", period_length)
    print("\t epochs =", epochs)
    print("\t max_len =", max_len)
    print("\t pool_window =", strides_1d)
    pass


# ==================================================
# 输出训练的结果
def output_result(results, predictions, y_test):
    print("模型预测-->", end = '')
    print("损失值 = {}，精确度 = {}".format(results[0], results[1]))
    if label_name == 'age':
        np_argmax = np.argmax(predictions, 1)
        print("前 30 个真实的目标数据 =", np.array(y_test[:30], dtype = int))
        print("前 30 个预测的目标数据 =", np.array(np.argmax(predictions[:30], 1), dtype = int))
        print("前 30 个预测的结果数据 =", )
        print(predictions[:30])
        for i in range(10):
            print("类别 {0} 的真实数目：{1}，预测数目：{2}".format(i, sum(y_test == i), sum(np_argmax == i)))
    elif label_name == 'gender':
        predict_gender = np.array(predictions > 0.5, dtype = int)
        print("sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) = error% =",
              sum(abs(predict_gender - y_test)) / sum(y_test) * 100, '%')
        print("前100个真实的目标数据 =", np.array(y_test[:100], dtype = int))
        print("前100个预测的目标数据 =", np.array(predict_gender[:100], dtype = int))
        print("sum(predictions>0.5) =", sum(predict_gender))
        print("sum(y_test) =", sum(y_test))
        print("sum(abs(predictions-y_test))=error_number=", sum(abs(predict_gender - y_test)))
    else:
        print("错误的标签名称：", label_name)
        pass
    pass


# ==================================================
def continue_train_model():
    # --------------------------------------------------
    print('>' * 15 + ' ' * 3 + "训练模型" + ' ' * 3 + '<' * 15)
    # --------------------------------------------------
    # 构建网络模型，输出模型相关的参数，构建多输入单输出模型，输出构建模型的结果
    print('-' * 5 + ' ' * 3 + "构建'{0}'模型".format(model_type) + ' ' * 3 + '-' * 5)
    # model=keras.models.load_model('/save_model/m0.h5')
    model = keras.models.load_model('/save_model/m1.h5')
    model.summary()
    print('-' * 5 + ' ' * 3 + "加载 npy 数据集" + ' ' * 3 + '-' * 5)
    X_train = np.load('save_data/fix_7_21_640k/x_train_' + label_name + '.npy', allow_pickle = True)
    y_train = np.load('save_data/fix_7_21_640k/y_train_' + label_name + '.npy', allow_pickle = True)
    X_test = np.load('save_data/fix_7_21_640k/x_test_' + label_name + '.npy', allow_pickle = True)
    y_test = np.load('save_data/fix_7_21_640k/y_test_' + label_name + '.npy', allow_pickle = True)


def train_model():
    # --------------------------------------------------
    print('>' * 15 + ' ' * 3 + "训练模型" + ' ' * 3 + '<' * 15)
    # --------------------------------------------------
    # 构建网络模型，输出模型相关的参数，构建多输入单输出模型，输出构建模型的结果
    print('-' * 5 + ' ' * 3 + "构建'{0}'模型".format(model_type) + ' ' * 3 + '-' * 5)
    output_parameters()
    model = construct_model()
    print(model.summary())
    print("保存原始：", model.save('save_model/m0.h5'))
    print('-' * 5 + ' ' * 3 + "素材数:{0}".format(creative_id_window) + ' ' * 3 + '-' * 5)
    # --------------------------------------------------
    print('-' * 5 + ' ' * 3 + "加载 npy 数据集" + ' ' * 3 + '-' * 5)
    X_train = np.load('save_data/fix_7_21_640k/x_train_' + label_name + '.npy', allow_pickle = True)
    y_train = np.load('save_data/fix_7_21_640k/y_train_' + label_name + '.npy', allow_pickle = True)
    X_test = np.load('save_data/fix_7_21_640k/x_test_' + label_name + '.npy', allow_pickle = True)
    y_test = np.load('save_data/fix_7_21_640k/y_test_' + label_name + '.npy', allow_pickle = True)

    # # 加载数据
    # print('-' * 5 + ' ' * 3 + "加载数据集" + ' ' * 3 + '-' * 5)
    # X_data, y_data = load_data()
    # output_example_data(X_data, y_data)
    # # --------------------------------------------------
    # # 清洗数据集，生成所需要的数据
    # print('-' * 5 + ' ' * 3 + "清洗数据集" + ' ' * 3 + '-' * 5)
    # X_doc, y_doc = generate_fix_data(X_data, y_data)
    # output_example_data(X_doc, y_doc)
    # del X_data, y_data  # 清空读取 csv 文件使用的内存
    # # --------------------------------------------------
    # # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # print('-' * 5 + ' ' * 3 + "拆分数据集" + ' ' * 3 + '-' * 5)
    # X_train, X_test, y_train, y_test = train_test_split(X_doc, y_doc, random_state = seed, stratify = y_doc)
    # print("训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))
    print('-' * 5 + ' ' * 3 + "训练数据集" + ' ' * 3 + '-' * 5)
    output_example_data(X_train, y_train)
    print('-' * 5 + ' ' * 3 + "测试数据集" + ' ' * 3 + '-' * 5)
    output_example_data(X_test, y_test)
    # --------------------------------------------------
    # 训练网络模型
    # 使用验证集
    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    history = model.fit({
        'creative_id': X_train[:, 0],
        'click_times': X_train[:, 1].reshape((-1, time_id_max * period_length // period_days, 1)),
    }, y_train, epochs = epochs, batch_size = batch_size,
        validation_split = 0.2, use_multiprocessing = True, verbose = 2)
    print("保存第一次训练模型", model.save_weights('save_model/m1.bin'))
    f = open('m2.pkl', 'wb')
    pickle.dump(history, f)
    f.close()
    results = model.evaluate({
        'creative_id': X_test[:, 0],
        'click_times': X_test[:, 1].reshape((-1, time_id_max * period_length // period_days, 1)),
    }, y_test, use_multiprocessing = True, verbose = 0)
    predictions = model.predict({
        'creative_id': X_test[:, 0],
        'click_times': X_test[:, 1].reshape((-1, time_id_max * period_length // period_days, 1)),
    }).squeeze()
    output_result(results, predictions, y_test)
    # --------------------------------------------------
    # 不使用验证集，训练次数减半
    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    history = model.fit({
        'creative_id': X_train[:, 0],
        'click_times': X_train[:, 1].reshape((-1, time_id_max * period_length // period_days, 1)),
    }, y_train, epochs = epochs // 2, batch_size = batch_size, use_multiprocessing = True,
        verbose = 2)
    print("保存第二次训练模型", model.save_weights('save_model/m2.bin'))
    f = open('m2.pkl', 'wb')
    pickle.dump(history, f)
    f.close()

    results = model.evaluate({
        'creative_id': X_test[:, 0],
        'click_times': X_test[:, 1].reshape((-1, time_id_max * period_length // period_days, 1)),
    }, y_test, use_multiprocessing = True, verbose = 0)
    predictions = model.predict({
        'creative_id': X_test[:, 0],
        'click_times': X_test[:, 1].reshape((-1, time_id_max * period_length // period_days, 1)),
    }).squeeze()
    output_result(results, predictions, y_test)
    pass


def train_age():
    global model_type, label_name
    model_type = ''
    label_name = 'age'  # age: 多分类问题；gender: 二分类问题
    train_model()


def train_gender():
    global model_type, label_name
    model_type = ''
    label_name = 'gender'  # age: 多分类问题；gender: 二分类问题
    train_model()


# ==================================================
if __name__ == '__main__':
    # 定义全局通用变量
    # file_name = './data/train_data_all_sequence_v_little.csv'
    # file_name = './data/train_data_all_sequence_v.csv'
    file_name = './data/train_data_all_min_complete_v.csv'
    model_type = ''
    label_name = ''
    age_sigmoid = -1
    # 定义全局序列变量
    user_id_num = 900000  # 用户数
    creative_id_max = 2481135 - 1  # 最大的素材编号 = 素材的总数量 - 1，这个编号已经修正了数据库与Python索引的区别
    time_id_max = 91
    click_times_max = 152  # 所有素材中最大的点击次数
    ad_id_max = 2264190  # 最大的广告编号=广告的种类
    product_id_max = 44313  # 最大的产品编号
    category_max = 18  # 最大的产品类别编号
    advertiser_id_max = 62965  # 最大的广告主编号
    industry_max = 335  # 最大的产业类别编号
    field_list = [
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]
    field_num = 4
    # 序列数据、循环数据、未知词、序列头 对于 GlobalMaxPooling1D() 函数都是没有用处的，因此默认关闭
    dictionary_asc = True  # 字典按正序(asc)取，还是倒序(desc)取
    unknown_word = False  # 是否使用未知词 1
    data_seq_head = False  # 是否生成序列头 2
    sequence_data = False  # 不使用序列数据，如果使用序列数据，则上两项默认为 True，即原始设置无用
    sequence_loop = False  # 序列数据是否循环生成，即把91天的访问数据再重复几遍填满 max_len，而不是使用 0 填充
    # 定义全局素材变量
    # 素材库大小 = creative_id_end - creative_id_start = creative_id_num = creative_id_step_size * (1 + 3 + 1)
    creative_id_step_size = 128000
    creative_id_window = creative_id_step_size * 5
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window
    # --------------------------------------------------
    # 定义全局模型变量
    batch_size = 512
    embedding_size = 128  # 嵌入维度
    period_days = 7
    period_length = 21  # 每个周期的素材数目
    # {1: 13922521, 2: 3704125, 3: 1239096, 4: 490834, 5: 216510, 6: 106288, 7: 56275, 8: 31523, 9: 18830, 10: 12162,}
    # {11	7726, 12	5014, 13	3446, 14	2472, 15	1763 , 16	1354, 17	970, 18	740, 19	544, 20	497}
    strides_1d = 7  # Conv1D的步长
    epochs = 20
    max_len = 128  # {64:803109，128:882952 个用户}  {64：1983350，128：2329077 个素材}
    # max_len = 256
    RMSProp_lr = 3e-04
    # 运行训练程序
    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1000)
    # train_age()
    train_gender()

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
import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import winsound
from keras import Input
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, LSTM
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential, Model

# --------------------------------------------------
from tensorflow.python.keras import losses, metrics, optimizers
from tensorflow.python.keras.layers import Activation, BatchNormalization, concatenate, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

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
    # "time_id","user_id_inc","user_id","creative_id_inc","creative_id","click_times","age","gender"
    df = pd.read_csv(file_name, dtype = int)
    # --------------------------------------------------
    # 输入数据处理：选择需要的列
    if sequence_data:
        X_csv = df[["user_id_inc", "creative_id_inc", "time_id", "click_times"]].values
        X_csv[:, 2] = X_csv[:, 2] - 1
    else:
        X_csv = df[["user_id_inc", "creative_id_inc"]].values
        pass
    # 索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的，因此数据都减去1
    # 没有在数据库中处理索引，是因为尽量不在数据库中修正原始数据，除非是不得不变更的数据，这样子业务逻辑清楚
    X_csv[:, 0] = X_csv[:, 0] - 1
    X_csv[:, 1] = X_csv[:, 1] - 1
    # --------------------------------------------------
    # 目标数据处理：目标字段的偏移量是 -1，是因为索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的
    # 既可以加载 'age'，也可以加载 'gender'
    y_csv = df[label_name].values - 1
    print("数据加载完成。")
    return X_csv, y_csv


# ==================================================
# 生成所需要的数据
def generate_data(X_data, y_data):
    global unknown_word, data_seq_head
    print("数据生成中：", end = '')
    y_doc = np.zeros([user_id_num], dtype = int)
    # 初始化 X_doc 为空的列表
    X_doc = np.zeros([user_id_num, 2], dtype = object)
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    tmp_user_id = -1
    tmp_time_id = 0
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
            pass
        user_id = row_data[0]
        time_id = row_data[2]
        y_doc[user_id] = y_data[i]
        # 整理过的数据已经按照 user_id 的顺序编号，当 user_id 变化时，就代表前一个用户的数据已经清洗完成
        if user_id != tmp_user_id:
            # 将前一个用户序列重复填充
            if tmp_user_id != -1:
                # NOTE: List 必须用拷贝，否则就是引用赋值，这个值就没有起到临时变量的作用，会不断改变
                tmp_creative_id_sequence = X_doc[tmp_user_id, 0].copy()
                if max_len > len(tmp_creative_id_sequence):
                    for j in range(max_len // len(tmp_creative_id_sequence)):
                        X_doc[tmp_user_id, 0].extend(tmp_creative_id_sequence)
                        pass
                    pass
                pass

            # 初始化新的用户序列
            X_doc[user_id, 0] = [2]  # 数据序列用 2 表示用户序列的开始
            X_doc[user_id, 1] = [2]  # 数据序列用 2 表示用户序列的开始
            tmp_user_id = user_id
            tmp_time_id = time_id
            pass

        # 如果生成的是序列数据，就需要更新 time_id, tmp_time_id
        if (time_id - tmp_time_id) > 1:  # 如果两个数据的时间间隔超过1天就需要插入一个 0
            X_doc[user_id, 0].append(0)
            X_doc[user_id, 1].append(0)
            pass
        tmp_time_id = time_id

        # 生成 creative_id
        # 0 表示 “padding”（填充），1 表示 “unknown”（未知词），2 表示 “start”（用户开始）
        # 'creative_id_inc' 字段的偏移量为 3，是因为需要保留 {0, 1, 2} 三个数：
        creative_id = row_data[1] + 3  # 词典是正序取，就是从小到大

        if creative_id_end > creative_id > creative_id_begin:
            creative_id = creative_id - creative_id_begin
        elif creative_id < creative_id_end - creative_id_max:
            creative_id = creative_id_max - creative_id_begin + creative_id
        elif unknown_word:
            creative_id = 1  # 超过词典大小的素材标注为 1，即「未知」
        else:
            continue
        X_doc[user_id, 0].append(creative_id)

        click_times = row_data[3]
        X_doc[user_id, 1].append(click_times)
        pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ==================================================
def output_example_data(X, y):
    print("数据(X[0], y[0]) =", X[0], y[0])
    print("数据(X[30], y[30]) =", X[30], y[30])
    print("数据(X[600], y[600]) =", X[600], y[600])
    if len(y) > 8999:
        print("数据(X[9000], y[9000]) =", X[9000], y[9000])
    if len(y) > 11999:
        print("数据(X[120000], y[120000]) =", X[120000], y[120000])
    if len(y) > 224999:
        print("数据(X[224999], y[224999]) =", X[224999], y[224999])
    if len(y) > 674999:
        print("数据(X[674999], y[674999]) =", X[674999], y[674999])
    if len(y) > 899999:
        print("数据(X[899999], y[899999]) =", X[899999], y[899999])
        pass
    pass


# ==================================================
# 训练网络模型
def construct_model():
    input_creative_id = Input(shape = (None,), dtype = 'int32', name = 'creative_id')
    embedded_creative_id = Embedding(creative_id_window, embedding_size)(input_creative_id)
    encoded_creative_id = GlobalMaxPooling1D()(embedded_creative_id)

    input_time_id = Input(shape = (None,), dtype = 'int32', name = 'time_id')
    encoded_time_id = LSTM(max_len)(input_time_id)

    concatenated = concatenate([encoded_creative_id, encoded_time_id], axis = -1)
    x = Dropout(0.5)(concatenated)
    x = Dense(embedding_size, kernel_regularizer = l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(concatenated)
    x = Dense(embedding_size, kernel_regularizer = l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if label_name == "age":
        x = Dropout(0.5)(x)
        x = Dense(10, kernel_regularizer = l2(0.001))(x)
        x = BatchNormalization()(x)
        output_tensor = Activation('softmax')(x)
        model = Model([input_creative_id, input_time_id], output_tensor)
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
                      loss = losses.sparse_categorical_crossentropy,
                      metrics = [metrics.sparse_categorical_accuracy])
    elif label_name == 'gender':
        x = Dropout(0.5)(x)
        x = Dense(1, kernel_regularizer = l2(0.001))(x)
        x = BatchNormalization()(x)
        output_tensor = Activation('sigmoid')(x)
        model = Model([input_creative_id, input_time_id], output_tensor)
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
                      loss = losses.binary_crossentropy,
                      metrics = [metrics.binary_accuracy])
    else:
        raise Exception("错误的标签类型！")
    return model


# ==================================================
def train_model(X_data, y_data):
    # 清洗数据集，生成所需要的数据
    print('-' * 5 + ' ' * 3 + "清洗数据集" + ' ' * 3 + '-' * 5)
    X_doc, y_doc = generate_data(X_data, y_data)
    output_example_data(X_doc, y_doc)
    # --------------------------------------------------
    print('-' * 5 + ' ' * 3 + "拆分数据集" + ' ' * 3 + '-' * 5)
    X_train, X_test, y_train, y_test = train_test_split(
            X_doc, y_doc, random_state = seed, stratify = y_doc)
    print("训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))
    # --------------------------------------------------
    # 填充数据集
    print('-' * 5 + ' ' * 3 + "填充数据集" + ' ' * 3 + '-' * 5)
    X_train_creative_id = pad_sequences(X_train[:, 0], maxlen = max_len, padding = 'post')
    X_test_creative_id = pad_sequences(X_test[:, 0], maxlen = max_len, padding = 'pose')
    X_train_click_times = pad_sequences(X_train[:, 1], maxlen = max_len, padding = 'post')
    X_test_click_times = pad_sequences(X_test[:, 1], maxlen = max_len, padding = 'post')
    # --------------------------------------------------
    # 构建模型
    print('-' * 5 + ' ' * 3 + "构建网络模型" + ' ' * 3 + '-' * 5)
    model = construct_model()
    print(model.summary())

    # --------------------------------------------------
    # 输出训练的结果
    def output_result():
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

    # --------------------------------------------------
    # 训练网络模型
    # 使用验证集
    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
              validation_split = 0.2, use_multiprocessing = True, verbose = 2)
    results = model.evaluate(X_test, y_test, verbose = 0)
    predictions = model.predict(X_test).squeeze()
    output_result()

    # --------------------------------------------------
    # 不使用验证集，训练次数减半
    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    model.fit(X_train, y_train, epochs = 10, batch_size = batch_size,
              use_multiprocessing = True, verbose = 2)
    results = model.evaluate(X_test, y_test, verbose = 0)
    predictions = model.predict(X_test).squeeze()
    output_result()


# ==================================================
def train_single_age():
    global file_name, model_type, label_name
    global max_len, embedding_size, sequence_data, sequence_loop, epochs, batch_size
    global creative_id_window, creative_id_begin, creative_id_end
    print('>' * 15 + ' ' * 3 + "train_single_age" + ' ' * 3 + '<' * 15)
    # --------------------------------------------------
    # 加载数据
    print('-' * 5 + ' ' * 3 + "加载数据集" + ' ' * 3 + '-' * 5)
    file_name = './data/train_data_all_sequence_v_little.csv'
    file_name = './data/train_data_all_sequence_v.csv'
    model_type = 'GlobalMaxPooling1D+MLP'
    label_name = 'gender'
    # --------------------------------------------------
    # 定义全局定制变量
    max_len = 256
    embedding_size = 128
    epochs = 60
    batch_size = 1024
    sequence_data = True
    sequence_loop = True
    # 定制 素材库大小
    creative_id_window = creative_id_step_size * 1
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window
    X_data, y_data = load_data()
    output_example_data(X_data, y_data)
    print('-' * 5 + ' ' * 3 + "素材数:{0}".format(creative_id_window) + ' ' * 3 + '-' * 5)
    train_model(X_data, y_data)
    pass


# ==================================================
if __name__ == '__main__':
    # 定义全局通用变量
    file_name = './data/train_data_all_no_sequence.csv'
    label_name = 'age'
    time_id_max = 91
    user_id_num = 900000  # 用户数
    creative_id_max = 2481135 - 1  # 最大的素材编号 = 素材的总数量 - 1，这个编号已经修正了数据库与Python索引的区别
    ad_id_max = 2264190  # 最大的广告编号=广告的种类
    product_id_max = 44313  # 最大的产品编号
    product_category_max = 18  # 最大的产品类别编号
    advertiser_id_max = 62965  # 最大的广告主编号
    industry_max = 335  # 最大的产业类别编号
    click_times_max = 152  # 所有素材中最大的点击次数
    model_type = "GlobalMaxPooling1D"
    RMSProp_lr = 5e-04
    epochs = 20
    batch_size = 256
    dictionary_asc = True  # 字典按正序(asc)取，还是倒序(desc)取
    # 序列数据、循环数据、未知词、序列头 对于 GlobalMaxPooling1D() 函数都是没有用处的，因此默认关闭
    unknown_word = False  # 是否使用未知词 1
    data_seq_head = False  # 是否生成序列头 2
    sequence_data = False  # 不使用序列数据，如果使用序列数据，则上两项默认为 True，即原始设置无用
    sequence_loop = False  # 序列数据是否循环生成，即把91天的访问数据再重复几遍填满 max_len，而不是使用 0 填充
    # --------------------------------------------------
    # 定义全局定制变量
    max_len = 128  # 64:803109，128:882952 个用户；64：1983350，128：2329077 个素材
    embedding_size = 128
    # 定制 素材库大小 = creative_id_end - creative_id_start = creative_id_num = creative_id_step_size * (1 + 3 + 1)
    creative_id_step_size = 128000
    creative_id_window = creative_id_step_size * 10
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window
    # --------------------------------------------------
    # 运行训练程序
    train_single_age()
    # train_single_gender()
    # train_multi_fraction()
    # train_batch_age()
    # train_batch_gender()
    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1000)
    plt.show()

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

from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras import losses, metrics, optimizers
from tensorflow.python.keras.layers import Activation, BatchNormalization, concatenate, Dropout, Bidirectional
from tensorflow.python.keras.layers import Dense, Embedding, GlobalMaxPooling1D, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.regularizers import l2

# from tensorflow_core.python.keras import Input
# from tensorflow_core.python.keras import losses, metrics, optimizers
# from tensorflow_core.python.keras.layers import Activation, BatchNormalization, concatenate, Dropout, Bidirectional
# from tensorflow_core.python.keras.layers import Dense, Embedding, GlobalMaxPooling1D, LSTM
# from tensorflow_core.python.keras.models import Model
# from tensorflow_core.python.keras.regularizers import l2
# from keras_preprocessing.sequence import pad_sequences

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
    # 索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的，因此数据都减去1
    # 没有在数据库中处理索引，是因为尽量不在数据库中修正原始数据，除非是不得不变更的数据，这样子业务逻辑清楚
    # user_id_inc:0, creative_id_inc:1, time_id:2, product_id:3, product_category:4, advertiser_id:5,industry:6
    # click_times:7 数据是值，不需要减 1
    for j in range(field_num - 1):
        X_csv[:, j] = X_csv[:, j] - 1
    # --------------------------------------------------
    # 目标数据处理：目标字段的偏移量是 -1，是因为索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的
    # 既可以加载 'age'，也可以加载 'gender'
    y_csv = df[label_name].values - 1
    print("数据加载完成。")
    return X_csv, y_csv


# ==================================================
# 生成所需要的数据
def generate_data(X_data, y_data):
    print("数据生成中：", end = '')
    y_doc = np.zeros([user_id_max], dtype = int)
    # 初始化 X_doc 为空的列表
    # X_doc[:,0]: creative_id
    # X_doc[:,1]: product_id, X_doc[:,2]: product_category
    # X_doc[:,3]: advertiser_id, X_doc[:,4]: industry
    # X_doc[:,5]: click_times
    # filed_list 去除 user_id, time_id
    train_field_num = field_num - 2
    X_doc = np.zeros([user_id_max, train_field_num], dtype = object)
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    prev_user_id = -1
    prev_time_id = 0
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
            pass
        user_id = row_data[0]
        time_id = row_data[2]
        if user_id >= user_id_max:
            break
        y_doc[user_id] = y_data[i] if age_sigmoid == -1 or label_name == 'gender' else int(age_sigmoid == y_data[i])
        # 整理过的数据已经按照 user_id 的顺序编号，当 user_id 变化时，就代表前一个用户的数据已经清洗完成
        if user_id != prev_user_id:
            # 将前一个用户序列重复填充
            if prev_user_id != -1:
                for j in range(train_field_num):
                    pre_data = X_doc[prev_user_id, j]
                    pre_data_len = len(pre_data)
                    if max_len > pre_data_len:
                        # NOTE: List 必须用拷贝，否则就是引用赋值，这个值就没有起到临时变量的作用，会不断改变
                        pre_data_copy = pre_data.copy()
                        for k in range(max_len // pre_data_len):
                            pre_data.extend(pre_data_copy)

            # 初始化新的用户序列, 数据序列用 2 表示用户序列的开始
            for j in range(train_field_num - 1):
                X_doc[user_id, j] = [2]
                pass
            X_doc[user_id, train_field_num - 1] = []
            # 重置临时变量
            prev_user_id = user_id
            prev_time_id = time_id
            pass

        # 如果生成的是序列数据，就需要更新 time_id, prev_time_id
        if (time_id - prev_time_id) > 1:  # 如果两个数据的时间间隔超过1天就需要插入一个 0
            for j in range(train_field_num):
                X_doc[user_id, j].append(0)

        prev_time_id = time_id

        # 生成 creative_id
        # 0 表示 “padding”（填充），1 表示 “unknown”（未知词），2 表示 “start”（用户开始）
        # 'creative_id_inc' 字段的偏移量为 3，是因为需要保留 {0, 1, 2} 三个数：
        # row_data[0]: user_id
        creative_id = row_data[1] + 3  # 词典是正序取，就是从小到大
        # row_data[2]: time_id
        product_id = row_data[3] + 3  # product_id 词库很小，因此不会产生未知词，保留 1 是保持概念统一
        category = row_data[4] + 3
        advertiser_id = row_data[5] + 3
        industry = row_data[6] + 3
        click_times = row_data[7]  # 这个不是词典，本身就是值，不需要再调整

        # 素材是关键，没有素材编号的数据全部放弃
        if creative_id_end > creative_id > creative_id_begin:
            creative_id = creative_id - creative_id_begin
        elif creative_id < creative_id_end - creative_id_max:
            creative_id = creative_id_max - creative_id_begin + creative_id
        elif unknown_word:
            creative_id = 1  # 超过词典大小的素材标注为 1，即「未知」
        else:
            continue

        X_doc[user_id, 0].append(creative_id)
        X_doc[user_id, 1].append(click_times)
        if product_id == 2:  # FIXME：这段代码以后可以在数据库调整数据后取消
            product_id = 1
        X_doc[user_id, 2].append(product_id)
        X_doc[user_id, 3].append(category)
        X_doc[user_id, 4].append(advertiser_id)
        if industry == 2:  # FIXME：这段代码以后可以在数据库调整数据后取消
            industry = 1
        X_doc[user_id, 5].append(industry)
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
# 训练网络模型
def construct_model():
    input_creative_id = Input(shape = (None,), dtype = 'int32', name = 'creative_id')
    embedded_creative_id = Embedding(
        creative_id_window, embedding_size, name = 'embedded_creative_id')(input_creative_id)
    encoded_creative_id = GlobalMaxPooling1D(name = 'encoded_creative_id')(embedded_creative_id)

    input_product_id = Input(shape = (None,), dtype = 'int32', name = 'product_id')
    embedded_product_id = Embedding(product_id_max, 32, name = 'embedded_product_id')(input_product_id)
    encoded_product_id = GlobalMaxPooling1D(name = 'encoded_product_id')(embedded_product_id)

    input_category = Input(shape = (None,), dtype = 'int32', name = 'category')
    embedded_category = Embedding(category_max, 2, name = 'embedded_category')(input_category)
    encoded_category = GlobalMaxPooling1D(name = 'encoded_category')(embedded_category)
    # encoded_category = Bidirectional(
    #     LSTM(32, dropout = 0.2, recurrent_dropout = 0.2), name = 'encoded_category')(embedded_category)

    input_advertiser_id = Input(shape = (None,), dtype = 'int32', name = 'advertiser_id')
    embedded_advertiser_id = Embedding(advertiser_id_max, 32, name = 'embedded_advertiser_id')(input_advertiser_id)
    encoded_advertiser_id = GlobalMaxPooling1D(name = 'encoded_advertiser_id')(embedded_advertiser_id)

    input_industry = Input(shape = (None,), dtype = 'int32', name = 'industry')
    embedded_industry = Embedding(industry_max, 16, name = 'embedded_industry')(input_industry)
    encoded_industry = GlobalMaxPooling1D(name = 'encoded_industry')(embedded_industry)
    # encoded_industry = Bidirectional(
    #     LSTM(32, dropout = 0.2, recurrent_dropout = 0.2, name = 'encoded_industry'))(embedded_industry)

    # LSTM(14) : 是因为 91 天正好是 14 个星期
    # LSTM(32) : 方便计算
    input_click_times = Input(shape = (None, 1), dtype = 'float32', name = 'click_times')
    encoded_click_times = Bidirectional(
        LSTM(32, dropout = 0.2, recurrent_dropout = 0.2), name = 'encoded_click_times')(input_click_times)

    concatenated = concatenate([
        encoded_creative_id,
        encoded_click_times,
        encoded_product_id,
        encoded_category,
        encoded_advertiser_id,
        encoded_industry
    ], axis = -1)

    x = Dropout(0.5, name = 'Dropout_0101')(concatenated)
    x = Dense(embedding_size, kernel_regularizer = l2(0.001), name = 'Dense_0101')(x)
    x = BatchNormalization(name = 'BN_0101')(x)
    x = Activation('relu', name = 'relu_0101')(x)

    x = Dropout(0.5, name = 'Dropout_0102')(x)
    x = Dense(embedding_size, kernel_regularizer = l2(0.001), name = 'Dense_0102')(x)
    x = BatchNormalization(name = 'BN_0102')(x)
    x = Activation('relu', name = 'relu_0102')(x)

    x = Dropout(0.5, name = 'Dropout_0103')(x)
    x = Dense(embedding_size, kernel_regularizer = l2(0.001), name = 'Dense_0103')(x)
    x = BatchNormalization(name = 'BN_0103')(x)
    x = Activation('relu', name = 'relu_0103')(x)

    x = Dropout(0.5, name = 'Dropout_0201')(x)
    x = Dense(embedding_size // 2, kernel_regularizer = l2(0.001), name = 'Dense_0201')(x)
    x = BatchNormalization(name = 'BN_0201')(x)
    x = Activation('relu', name = 'relu_0201')(x)

    x = Dropout(0.5, name = 'Dropout_0202')(x)
    x = Dense(embedding_size // 2, kernel_regularizer = l2(0.001), name = 'Dense_0202')(x)
    x = BatchNormalization(name = 'BN_0202')(x)
    x = Activation('relu', name = 'relu_0202')(x)

    x = Dropout(0.5, name = 'Dropout_0203')(x)
    x = Dense(embedding_size // 2, kernel_regularizer = l2(0.001), name = 'Dense_0203')(x)
    x = BatchNormalization(name = 'BN_0203')(x)
    x = Activation('relu', name = 'relu_0203')(x)

    if label_name == "age" and age_sigmoid == -1:
        x = Dropout(0.5)(x)
        x = Dense(10, kernel_regularizer = l2(0.001), name = 'output')(x)
        x = BatchNormalization()(x)
        output_tensor = Activation('softmax')(x)

        model = Model([
            input_creative_id,
            input_click_times,
            input_product_id,
            input_category,
            input_advertiser_id,
            input_industry
        ], output_tensor)

        print('-' * 5 + ' ' * 3 + "编译模型" + ' ' * 3 + '-' * 5)

        model.compile(optimizer = optimizers.RMSprop(lr = RMSProp_lr),
                      loss = losses.sparse_categorical_crossentropy,
                      metrics = [metrics.sparse_categorical_accuracy])
    elif label_name == 'gender' or age_sigmoid != -1:
        x = Dropout(0.5)(x)
        x = Dense(1, kernel_regularizer = l2(0.001), name = 'output')(x)
        x = BatchNormalization()(x)
        output_tensor = Activation('sigmoid')(x)

        model = Model([
            input_creative_id,
            input_click_times,
            input_product_id,
            input_category,
            input_advertiser_id,
            input_industry
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
    print("\tuser_id_maxber =", user_id_max)
    print("\tcreative_id_max =", creative_id_max)
    print("\tcreative_id_step_size =", creative_id_step_size)
    print("\tcreative_id_window =", creative_id_window)
    print("\tcreative_id_begin =", creative_id_begin)
    print("\tcreative_id_end =", creative_id_end)
    print("\tmax_len =", max_len)
    print("\tembedding_size =", embedding_size)
    print("\tepochs =", epochs)
    print("\tbatch_size =", batch_size)
    print("\tRMSProp =", RMSProp_lr)
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
def train_model():
    # --------------------------------------------------
    print('>' * 15 + ' ' * 3 + "训练模型" + ' ' * 3 + '<' * 15)
    # --------------------------------------------------
    # 构建网络模型，输出模型相关的参数，构建多输入单输出模型，输出构建模型的结果
    print('-' * 5 + ' ' * 3 + "构建'{0}'模型".format(model_type) + ' ' * 3 + '-' * 5)
    output_parameters()
    model = construct_model()
    print(model.summary())
    print('-' * 5 + ' ' * 3 + "素材数:{0}".format(creative_id_window) + ' ' * 3 + '-' * 5)
    # --------------------------------------------------
    # 加载数据
    print('-' * 5 + ' ' * 3 + "加载数据集" + ' ' * 3 + '-' * 5)
    X_data, y_data = load_data()
    output_example_data(X_data, y_data)
    # --------------------------------------------------
    # 清洗数据集，生成所需要的数据
    print('-' * 5 + ' ' * 3 + "清洗数据集" + ' ' * 3 + '-' * 5)
    X_doc, y_doc = generate_data(X_data, y_data)
    output_example_data(X_doc, y_doc)
    del X_data, y_data  # 清空读取 csv 文件使用的内存
    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    print('-' * 5 + ' ' * 3 + "拆分数据集" + ' ' * 3 + '-' * 5)
    X_doc_user_id = np.arange(0, user_id_max)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        X_doc_user_id, y_doc, random_state = seed, stratify = y_doc)
    print("训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))
    # --------------------------------------------------
    # 截断与填充数据集，按照 max_len 设定长度，将不足的数据集填充 0， 将过长的数据集截断
    print('-' * 5 + ' ' * 3 + "填充数据集" + ' ' * 3 + '-' * 5)
    X_train_creative_id = pad_sequences(X_doc[X_train_idx, 0], maxlen = max_len, padding = 'post')
    X_test_creative_id = pad_sequences(X_doc[X_test_idx, 0], maxlen = max_len, padding = 'post')
    output_example_data(X_train_creative_id, X_test_creative_id)

    X_train_product_id = pad_sequences(X_doc[X_train_idx, 2], maxlen = max_len, padding = 'post')
    X_test_product_id = pad_sequences(X_doc[X_test_idx, 2], maxlen = max_len, padding = 'post')
    output_example_data(X_train_product_id, X_test_product_id)

    X_train_category = pad_sequences(X_doc[X_train_idx, 3], maxlen = max_len, padding = 'post')
    X_test_category = pad_sequences(X_doc[X_test_idx, 3], maxlen = max_len, padding = 'post')
    output_example_data(X_train_category, X_test_category)

    X_train_advertiser_id = pad_sequences(X_doc[X_train_idx, 4], maxlen = max_len, padding = 'post')
    X_test_advertiser_id = pad_sequences(X_doc[X_test_idx, 4], maxlen = max_len, padding = 'post')
    output_example_data(X_train_advertiser_id, X_test_advertiser_id)

    X_train_industry = pad_sequences(X_doc[X_train_idx, 5], maxlen = max_len, padding = 'post')
    X_test_industry = pad_sequences(X_doc[X_test_idx, 5], maxlen = max_len, padding = 'post')
    output_example_data(X_train_industry, X_test_industry)

    X_train_click_times = pad_sequences(
        X_doc[X_train_idx, 1],
        maxlen = max_len,
        padding = 'post'
    ).reshape((-1, max_len, 1))  # -1 表示原始维度不变
    X_test_click_times = pad_sequences(
        X_doc[X_test_idx, 1],
        maxlen = max_len,
        padding = 'post'
    ).reshape((-1, max_len, 1))
    output_example_data(X_train_click_times, X_test_click_times)

    # --------------------------------------------------
    # 训练网络模型
    # 使用验证集
    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    model.fit({
        'creative_id': X_train_creative_id,
        'click_times': X_train_click_times,
        'product_id': X_train_product_id,
        'category': X_train_category,
        'advertiser_id': X_train_advertiser_id,
        'industry': X_train_industry
    }, y_train, epochs = epochs, batch_size = batch_size,
        validation_split = 0.2, use_multiprocessing = True, verbose = 2)
    results = model.evaluate({
        'creative_id': X_test_creative_id,
        'click_times': X_test_click_times,
        'product_id': X_test_product_id,
        'category': X_test_category,
        'advertiser_id': X_test_advertiser_id,
        'industry': X_test_industry
    }, y_test, use_multiprocessing = True, verbose = 0)
    predictions = model.predict({
        'creative_id': X_test_creative_id,
        'click_times': X_test_click_times,
        'product_id': X_test_product_id,
        'category': X_test_category,
        'advertiser_id': X_test_advertiser_id,
        'industry': X_test_industry
    }).squeeze()
    output_result(results, predictions, y_test)

    # --------------------------------------------------
    # 不使用验证集，训练次数减半
    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    model.fit({
        'creative_id': X_train_creative_id,
        'click_times': X_train_click_times,
        'product_id': X_train_product_id,
        'category': X_train_category,
        'advertiser_id': X_train_advertiser_id,
        'industry': X_train_industry
    }, y_train, epochs = epochs // 2, batch_size = batch_size, use_multiprocessing = True,
        verbose = 2)
    results = model.evaluate({
        'creative_id': X_test_creative_id,
        'click_times': X_test_click_times,
        'product_id': X_test_product_id,
        'category': X_test_category,
        'advertiser_id': X_test_advertiser_id,
        'industry': X_test_industry
    }, y_test, use_multiprocessing = True, verbose = 0)
    predictions = model.predict({
        'creative_id': X_test_creative_id,
        'click_times': X_test_click_times,
        'product_id': X_test_product_id,
        'category': X_test_category,
        'advertiser_id': X_test_advertiser_id,
        'industry': X_test_industry
    }).squeeze()
    output_result(results, predictions, y_test)
    pass


# ==================================================
if __name__ == '__main__':
    # 定义全局通用变量
    # file_name = './data/train_data_all_sequence_v_little.csv'
    file_name = './data/train_data_all_sequence_v.csv'
    model_type = 'GlobalMaxPooling1D+MLP'
    label_name = 'age'  # age: 多分类问题；gender: 二分类问题
    age_sigmoid = 2  # age_sigmoid==-1: 多分类问题，否则是对某个类别的二分类问题
    # 定义全局序列变量
    user_id_max = 900000  # 用户数
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
        "product_id",  # 3
        "product_category",  # 4
        "advertiser_id",  # 5
        "industry",  # 6
        "click_times",  # 7, click_times 属于值，不属于编号，不能再减1
    ]
    field_num = 8
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
    batch_size = 1024
    embedding_size = 128
    epochs = 30
    max_len = 128  # {64:803109，128:882952 个用户}  {64：1983350，128：2329077 个素材}
    # max_len = 256
    RMSProp_lr = 5e-04
    # 运行训练程序
    train_model()
    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1000)

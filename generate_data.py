# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
--------------------------------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   generate_data.py
@Version    :   v0.1
@Time       :   2020-06-07 15:41
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :   数据生成模块
@理解：
"""
import random

import numpy as np

import config
from config import creative_id_begin, creative_id_end
from config import creative_id_max, time_id_max, user_id_max
from show_data import show_example_data
from tools import show_title


# ----------------------------------------------------------------------
def generate_w2v_data(x_data):
    show_title("生成用于训练嵌入式模型的数据")
    data_length = user_id_max
    data_step = data_length // 100
    w2v_data = np.empty([user_id_max], dtype=object)
    for i in range(user_id_max):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
        user_list = w2v_data[i] = []
        for day_list in x_data[i]:
            if day_list is not None:
                for k in day_list:
                    user_list.extend(chr(k))
    print("第 {0} 条数据-->数据清洗完成。".format(i))
    return w2v_data


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
def generate_day_list_data(X_csv, y_csv):
    """
    生成每个用户每天访问数据的不截断列表，用于生成每天的均值数据(或者其他按天给出的统计特征)
    :param X_csv: 待训练使用的原始数据
    :param y_csv:
    """
    show_title('生成每个用户每天访问数据的不截断列表')
    data_length = y_csv.shape[0]
    data_step = data_length // 100
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    X_creative_id = np.empty([user_id_max, time_id_max], dtype=object)
    creative_id_list = X_creative_id[0, 0]
    # X_product_category = np.empty([user_id_max, time_id_max,1], dtype=object)
    y_gender = np.empty([user_id_max], dtype=int)
    prev_user_id = prev_time_id = -1
    for i, row_data in enumerate(X_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]
        time_id = row_data[2]
        click_times = row_data[3]
        # product_category = row_data[4]

        if user_id > prev_user_id:
            y_gender[user_id] = y_csv[i]
            prev_user_id = user_id
            prev_time_id = -1

        if time_id > prev_time_id:
            prev_time_id = time_id
            creative_id_list = X_creative_id[user_id, time_id] = []

        for _ in range(click_times):
            creative_id_list.append(creative_id)
            # X_creative_id[user_id, time_id, 1].append(product_category)
    print("第 {0} 条数据-->数据清洗完成。".format(i))
    return np.array(X_creative_id), np.array(y_gender)


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
def generate_no_time_data(X_csv, y_csv, train_field_num, label_field_num, repeat_creative_id):
    # TODO: 可以与 generate_day_list_data() 重构，将整理后数据的格式进行整理
    print('-' * 5 + "   清洗数据集   " + '-' * 5)
    data_length = y_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    X_doc = np.zeros([user_id_max, train_field_num], dtype=object)
    y_doc = np.zeros([user_id_max, label_field_num], dtype=int)
    # -1 不在数据序列中
    prev_user_id = -1
    for i, row_data in enumerate(X_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]

        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
        if user_id > prev_user_id:
            prev_user_id = user_id
            # 新建用户序列时，数据序列用 2 表示用户序列的开始，标签序列更新为用户的标签
            X_doc[user_id] = [2]  # 使用「序列开始标志」，统一数据格式
            for j in range(train_field_num):
                X_doc[user_id, j] = []

            for j in range(label_field_num):
                y_doc[user_id, j] = int(y_csv[i, j])

            pass

        # row_data[0]: user_id
        creative_id = row_data[1]
        # row_data[2]: time_id
        click_times = row_data[3]

        # 素材(creative_id)是关键
        if creative_id_end > creative_id > creative_id_begin:
            creative_id = creative_id - creative_id_begin
        elif creative_id < creative_id_end - creative_id_max:
            creative_id = creative_id_max - creative_id_begin + creative_id
        else:
            creative_id = 1  # 超过词典大小的素材标注为 1，即「未知」
            pass

        if X_doc[user_id, 0].count(
                creative_id) and not repeat_creative_id:  # 重复访问的素材不加入 no_time 的序列数据中
            continue

        X_doc[user_id, 0].append(creative_id)
        X_doc[user_id, 1].append(click_times)

    print("\n数据清洗完成！")
    show_example_data(X_doc, y_doc, data_type='清洗数据')
    return X_doc, y_doc


# ----------------------------------------------------------------------
# 生成没有时间间隔的数据
def generate_data_no_interval_with_repeat(x_csv, y_csv):
    # TODO: 可以与 generate_day_list_data() 重构，将整理后数据的格式进行整理
    print("数据生成中：", end='')
    # creative_id_list = [2]  # 第一个用户序列的起始标记，这个序列用于训练分类模型
    # word2vec_list = [chr(2)]  # 第一个用户序列的起始标记，这个序列用于训练嵌入数据
    # product_category_list = [0]  # 第一个用户序列的起始标记，针对creative_id的2赋予的0标记
    prev_user_id = x_csv[0, 0]  # 第一个用户的 user_id
    creative_id_list, product_category_list, word2vec_list = [], [], []  # 第一个用户的访问 sequence
    y_label = y_csv[0]  # 第一个用户的 label
    X_doc, y_doc, X_w2v = [[creative_id_list, product_category_list]], [y_label], [word2vec_list]  # 所有用户序列

    data_step = x_csv.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(x_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]
        click_times = row_data[3]
        product_category = row_data[4]

        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，新建一个新的用户序列并且添加到整个序列中
        if user_id > prev_user_id:
            prev_user_id = user_id

            creative_id_list, word2vec_list, product_category_list = [], [], []  # 第一个用户序列
            X_doc.append([creative_id_list, product_category_list])
            y_doc.append(y_csv[i])
            X_w2v.append(word2vec_list)
            pass

        for _ in range(click_times):
            creative_id_list.append(creative_id)
            word2vec_list.append(chr(creative_id))
            product_category_list.append(product_category)
        pass
    print('\t')
    return np.array(X_doc), np.array(y_doc), np.array(X_w2v)


def generate_balance_data(x_data, y_data):
    # TODO: 数据增强？
    if config.label_name == 'age':
        balance_list = config.balance_age_list
    elif config.label_name == 'gender':
        balance_list = config.balance_gender_list
    else:
        raise Exception("错误的标签类型！")

    x_extend, y_extend = [], []

    data_shape = x_data.shape[0] - 1
    data_step = data_shape // 100  # 标识数据清洗进度的步长
    for i, data_list in enumerate(x_data):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
        for j in range(balance_list[y_data[i]] - 1):
            # 搜索相同的用户序列进行混合
            data_index = random.randint(0, data_shape)
            while y_data[i] != y_data[data_index]:
                data_index = random.randint(0, data_shape)

            # 1. creative_id
            # 2. product_category
            tmp = data_list.copy()
            for k, t in zip(tmp, x_data[data_index]):
                k.extend(t)
            x_extend.append(tmp)
            y_extend.append(y_data[i])
    print('\t')
    return np.append(x_data, np.array(x_extend), axis=0), np.append(y_data, y_extend)

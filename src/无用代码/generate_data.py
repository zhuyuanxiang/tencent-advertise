# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   generate_data.py
@Version    :   v0.1
@Time       :   2020-08-17 15:48
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import math
import random

import numpy as np

import config
from config import creative_id_begin, creative_id_end, user_id_max, creative_id_max, time_id_max, fix_period_days, \
    fix_period_length
from src.data.show_data import show_example_data


def generate_word2vec_data_with_interval(x_csv):
    print('-' * 5 + "   清洗数据集:{0}~{1} 个素材   ".format(creative_id_begin, creative_id_end) + '-' * 5)
    data_length = x_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    x_creative_id = [[chr(2)] for _ in range(user_id_max)]
    prev_user_id = -1  # -1 不在数据序列中
    prev_time_id = -1  # 0 表示第 1 天
    time_id_interval = 0
    creative_id_list = None
    for i, row_data in enumerate(x_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        time_id = row_data[2]
        # 再次访问素材的间隔时间，间隔1天不插入0，间隔2天插入1个0，间隔3天插入2个0，间隔3天以上插入3个0
        time_id_interval = time_id - prev_time_id - 1
        time_id_interval = time_id_interval if time_id_interval < 3 else 3

        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
        # 没有修改代码为更为简洁和通用的形式，是现在操作速度会更快
        if user_id > prev_user_id:
            prev_user_id = user_id
            prev_time_id = 0
            time_id_interval = 0
            creative_id_list = x_creative_id[
                user_id]  # 这个是浅拷贝，即地址拷贝，修改 creative_id_list 即能修改 x_creative_id[user_id]
            pass

        creative_id_list.extend([chr(0) for _ in range(time_id_interval)])

        # row_data[0]: user_id
        creative_id = row_data[1]
        # row_data[2]: time_id

        # 素材(creative_id)是关键
        if creative_id_end > creative_id > creative_id_begin:
            creative_id = creative_id - creative_id_begin
        elif creative_id < creative_id_end - creative_id_max:
            creative_id = creative_id_max - creative_id_begin + creative_id
        else:
            creative_id = 1  # 超过词典大小的素材标注为 1，即「未知」
            pass

        creative_id_list.append(chr(creative_id))
        prev_time_id = time_id
    return x_creative_id


def generate_word2vec_data_no_interval(x_csv):
    print('-' * 5 + "   清洗数据集:{0}~{1} 个素材   ".format(creative_id_begin, creative_id_end) + '-' * 5)
    data_length = x_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    x_creative_id = [[chr(2)] for _ in range(user_id_max)]  # 每个序列都以 2
    prev_user_id = -1  # -1 不在数据序列中
    creative_id_list = None
    for i, row_data in enumerate(x_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]

        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
        # 没有修改代码为更为简洁和通用的形式，是现在操作速度会更快
        if user_id > prev_user_id:
            prev_user_id = user_id
            creative_id_list = x_creative_id[
                user_id]  # 这个是浅拷贝，即地址拷贝，修改 creative_id_list 即能修改 x_creative_id[user_id]
            pass

        # row_data[0]: user_id
        creative_id = row_data[1]
        # row_data[2]: time_id

        # 素材(creative_id)是关键
        if creative_id_end > creative_id > creative_id_begin:
            creative_id = creative_id - creative_id_begin
        elif creative_id < creative_id_end - creative_id_max:
            creative_id = creative_id_max - creative_id_begin + creative_id
        else:
            creative_id = 1  # 超过词典大小的素材标注为 1，即「未知」
            pass

        creative_id_list.append(chr(creative_id))
    return x_creative_id


def generate_fix_data(X_csv, y_csv):
    # TODO: 可以与 generate_day_list_data() 重构，将整理后的数据按照要求周期要求进行整理
    print('-' * 5 + "   清洗数据集   " + '-' * 5)
    data_length = y_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    # 初始化 X_doc 为空的列表 : X_doc[:,0]: creative_id, X_doc[:,1]: click_times
    # 初始化 y_doc 为空的列表 : y_doc[:,0]: age, X_doc[:,1]: gender
    # X_doc 的列表维度 = 去除 user_id, time_id
    X_doc = np.zeros([user_id_max, math.ceil(time_id_max / fix_period_days) * fix_period_length], dtype=object)
    y_doc = np.zeros([user_id_max, ], dtype=int)
    prev_user_id = -1  # 前一条数据的用户编号
    prev_period_index = 0  # 周期索引，即处理到第几个周期
    period_data_index = 0  # 周期内数据保存的位置索引
    for i, row_data in enumerate(X_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]
        time_id = row_data[2]
        click_times = row_data[3]

        # 整理过的数据已经按照 user_id 的顺序编号，当 user_id 变化时，就代表前一个用户的数据已经清洗完成
        if user_id > prev_user_id:
            prev_user_id = user_id
            prev_period_index = period_data_index = 0

        if (time_id - prev_period_index) >= fix_period_days:
            prev_period_index = time_id // fix_period_days * fix_period_days

        for _ in range(click_times):
            if period_data_index < fix_period_length:  # 每个周期访问素材填满后不再填充
                X_doc[user_id, prev_period_index + period_data_index] = creative_id
                period_data_index = period_data_index + 1

    print("\n数据清洗完成！")
    show_example_data(X_doc, y_doc, data_type='清洗数据')
    return X_doc, y_doc


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

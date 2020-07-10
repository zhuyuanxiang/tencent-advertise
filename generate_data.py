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
import math

import numpy as np
from sklearn.model_selection import train_test_split

from config import creative_id_max, time_id_max, user_id_num
from show_data import show_example_data


# ----------------------------------------------------------------------
def generate_word2vec_data_with_interval(x_csv, creative_id_begin, creative_id_end):
    print('-' * 5 + "   清洗数据集:{0}~{1} 个素材   ".format(creative_id_begin, creative_id_end) + '-' * 5)
    data_length = x_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    x_creative_id = [[chr(2)] for _ in range(user_id_num)]
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
            creative_id_list = x_creative_id[user_id]  # 这个是浅拷贝，即地址拷贝，修改 creative_id_list 即能修改 x_creative_id[user_id]
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


# ----------------------------------------------------------------------
def generate_word2vec_data_no_interval(x_csv, creative_id_begin, creative_id_end):
    print('-' * 5 + "   清洗数据集:{0}~{1} 个素材   ".format(creative_id_begin, creative_id_end) + '-' * 5)
    data_length = x_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    x_creative_id = [[chr(2)] for _ in range(user_id_num)]  # 每个序列都以 2
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
            creative_id_list = x_creative_id[user_id]  # 这个是浅拷贝，即地址拷贝，修改 creative_id_list 即能修改 x_creative_id[user_id]
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


# ----------------------------------------------------------------------
def generate_fix_data(X_csv, y_csv, train_field_num, label_field_num, period_length, period_days, creative_id_begin, creative_id_end):
    print('-' * 5 + "   清洗数据集   " + '-' * 5)
    data_length = y_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    # 初始化 X_doc 为空的列表 : X_doc[:,0]: creative_id, X_doc[:,1]: click_times
    # 初始化 y_doc 为空的列表 : y_doc[:,0]: age, X_doc[:,1]: gender
    # X_doc 的列表维度 = 去除 user_id, time_id
    X_doc = np.zeros([user_id_num, train_field_num, math.ceil(time_id_max / period_days) * period_length], dtype=object)
    y_doc = np.zeros([user_id_num, label_field_num], dtype=int)
    prev_user_id = -1
    prev_time_id = 0
    period_index = 0
    for i, row_data in enumerate(X_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        time_id = row_data[2]

        for j in range(label_field_num):
            y_doc[user_id, j] = y_csv[i, j]

        # 整理过的数据已经按照 user_id 的顺序编号，当 user_id 变化时，就代表前一个用户的数据已经清洗完成
        if user_id > prev_user_id:
            # 重置临时变量
            prev_user_id = user_id
            prev_time_id = 0
            period_index = 0
            pass

        if (time_id - prev_time_id) >= period_days:
            prev_time_id = time_id // period_days * period_days
            period_index = 0

        if period_index == period_length:  # 每个周期访问素材填满后不再填充
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
    show_example_data(X_doc, y_doc, data_type='清洗数据')
    return X_doc, y_doc


# ----------------------------------------------------------------------
def generate_no_time_data(X_csv, y_csv, train_field_num, label_field_num, repeat_creative_id, creative_id_begin, creative_id_end):
    print('-' * 5 + "   清洗数据集   " + '-' * 5)
    data_length = y_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    X_doc = np.zeros([user_id_num, train_field_num], dtype=object)
    y_doc = np.zeros([user_id_num, label_field_num], dtype=int)
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

        if X_doc[user_id, 0].count(creative_id) and not repeat_creative_id:  # 重复访问的素材不加入 no_time 的序列数据中
            continue

        X_doc[user_id, 0].append(creative_id)
        X_doc[user_id, 1].append(click_times)

    print("\n数据清洗完成！")
    show_example_data(X_doc, y_doc, data_type='清洗数据')
    return X_doc, y_doc


# ----------------------------------------------------------------------
def split_data(x_data, y_data, label_name):
    print('-' * 5 + "   拆分{0}数据集   ".format(label_name) + '-' * 5)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=seed, stratify=y_data)
    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------
# 生成没有时间间隔的数据
def generate_data_no_interval(X_data, y_data, creative_id_begin, creative_id_end):
    print("数据生成中：", end='')
    # 初始化 X_doc 为空的列表
    X_doc = [[2] for _ in range(user_id_num)]
    y_doc = [0 for _ in range(user_id_num)]
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    prev_user_id = -1  # -1 不在数据序列中
    creative_id_list = None
    for i, row_data in enumerate(X_data):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]
        y_doc[user_id] = y_data[i]
        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
        # 没有修改代码为更为简洁和通用的形式，是现在操作速度会更快
        if user_id > prev_user_id:
            prev_user_id = user_id
            creative_id_list = X_doc[user_id]  # 这个是浅拷贝，即地址拷贝，修改 creative_id_list 即能修改 x_creative_id[user_id]
            pass

        if creative_id_end > creative_id > creative_id_begin:
            creative_id = creative_id - creative_id_begin
        elif creative_id < creative_id_end - creative_id_max:
            creative_id = creative_id_max - creative_id_begin + creative_id
        else:
            creative_id = 1  # 超过词典大小的素材标注为 1，即「未知」

        creative_id_list.append(creative_id)
        pass
    print("\n数据清洗完成！")
    return X_doc, np.array(y_doc)

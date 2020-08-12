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
import config
import math
import numpy as np

from config import creative_id_max, time_id_max, user_id_max, fix_period_days, fix_period_length
from config import creative_id_window, creative_id_begin, creative_id_end
from config import seed
from show_data import show_example_data
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
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
def generate_day_list_data(X_csv, y_csv):
    """
    生成每个用户每天访问数据列表，用于生成每天的均值数据(或者其他按天给出的统计特征)
    :param X_csv:
    :param y_csv:
    """
    from tools import show_title
    show_title('清洗数据集')
    data_length = y_csv.shape[0]
    data_step = data_length // 100
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    X_doc = np.empty([user_id_max, time_id_max], dtype=object)
    y_doc = np.empty([user_id_max, ], dtype=int)
    prev_user_id = -1
    creative_id_list = None
    for i, row_data in enumerate(X_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]
        time_id = row_data[2]

        creative_id_list = X_doc[user_id, time_id]


# ----------------------------------------------------------------------
def generate_fix_data(X_csv, y_csv):
    print('-' * 5 + "   清洗数据集   " + '-' * 5)
    data_length = y_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end='')
    # 初始化 X_doc 为空的列表 : X_doc[:,0]: creative_id, X_doc[:,1]: click_times
    # 初始化 y_doc 为空的列表 : y_doc[:,0]: age, X_doc[:,1]: gender
    # X_doc 的列表维度 = 去除 user_id, time_id
    X_doc = np.zeros([user_id_max, math.ceil(time_id_max / fix_period_days) * fix_period_length], dtype=object)
    y_doc = np.zeros([user_id_max, ], dtype=int)
    prev_user_id = 0  # 前一条数据的用户编号
    prev_time_id = 0  # 前一条数据的日期编号
    prev_period_index = 0  # 周期索引，即处理到第几个周期
    period_data_index = 0  # 周期内数据保存的位置索引
    for i, row_data in enumerate(X_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        time_id = row_data[2]

        # 整理过的数据已经按照 user_id 的顺序编号，当 user_id 变化时，就代表前一个用户的数据已经清洗完成
        if user_id > prev_user_id:
            # 重置临时变量
            prev_user_id = user_id
            prev_time_id = 0
            period_data_index = 0
            pass

        if (time_id - prev_period_index) >= fix_period_days:
            prev_period_index = time_id // fix_period_days * fix_period_days

        if period_data_index < fix_period_length:  # 每个周期访问素材填满后不再填充
            # creative_id 的值已经在读取时修正过，增加了偏移量 3，保留了 {0,1,2}
            #   数据在 SQL 阶段已经处理了，不在字典中的数据已经置为1
            #   2 这个值在这个序列中没有使用，但是为了保持数据处理的统一性，没有修改
            # row_data[0]: user_id
            creative_id = row_data[1]
            # row_data[2]: time_id
            click_times = row_data[3]  # 这个不是词典，本身就是值，不需要再调整

            for _ in range(click_times):
                X_doc[user_id, prev_period_index + period_data_index] = creative_id
                period_data_index = period_data_index + 1

    print("\n数据清洗完成！")
    show_example_data(X_doc, y_doc, data_type='清洗数据')
    return X_doc, y_doc


# ----------------------------------------------------------------------
def generate_no_time_data(X_csv, y_csv, train_field_num, label_field_num, repeat_creative_id):
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

        if X_doc[user_id, 0].count(creative_id) and not repeat_creative_id:  # 重复访问的素材不加入 no_time 的序列数据中
            continue

        X_doc[user_id, 0].append(creative_id)
        X_doc[user_id, 1].append(click_times)

    print("\n数据清洗完成！")
    show_example_data(X_doc, y_doc, data_type='清洗数据')
    return X_doc, y_doc


# ----------------------------------------------------------------------
# 生成没有时间间隔的数据
def generate_data_no_interval_with_repeat(x_csv, y_csv):
    print("数据生成中：", end='')
    prev_user_id = x_csv[0, 0]  # 第一个用户的 user_id
    creative_id_list = [2]  # 第一个用户序列的起始标记
    word2vec_list = [chr(2)]  # 第一个用户序列的起始标记
    X_doc = [creative_id_list]
    X_w2v = [word2vec_list]
    y_doc = [y_csv[0]]  # 第一个用户的 label
    data_step = x_csv.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(x_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]

        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，新建一个新的用户序列并且添加到整个序列中
        if user_id > prev_user_id:
            prev_user_id = user_id
            creative_id_list = [2]
            X_doc.append(creative_id_list)
            word2vec_list = [chr(2)]
            X_w2v.append(word2vec_list)
            y_doc.append(y_csv[i])
            pass

        creative_id_list.append(creative_id)
        word2vec_list.append(chr(creative_id))
        pass
    print("\n数据清洗完成！")
    return np.array(X_doc), np.array(y_doc), np.array(X_w2v)


def generate_balance_data(x_data, y_data):
    if config.label_name == 'age':
        balance_list = config.balance_age_list
    elif config.label_name == 'gender':
        balance_list = config.balance_gender_list
    else:
        raise Exception("错误的标签类型！")

    x_extend, y_extend = [], []
    data_step = x_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, creative_id_list in enumerate(x_data):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end=';')
        for j in range(balance_list[y_data[i]]):
            x_extend.append(creative_id_list)
            y_extend.append(y_data[i])

    return np.append(x_data, x_extend), np.append(y_data, y_extend)


# ----------------------------------------------------------------------
def split_data(x_data, y_data, label_name):
    print('-' * 5 + "   拆分{0}数据集   ".format(label_name) + '-' * 5)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=seed, stratify=y_data)
    return x_train, y_train, x_test, y_test

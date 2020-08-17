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

import numpy as np

import config
import tools
from tensorflow import keras

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
from config import creative_id_begin, creative_id_end, user_id_max, creative_id_max, time_id_max, fix_period_days, \
    fix_period_length
from show_data import show_example_data

if __name__ == '__main__':
    # 运行结束的提醒
    tools.beep_end()


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

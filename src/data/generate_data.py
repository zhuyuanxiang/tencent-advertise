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

from config import embedding_size
from config import fix_period_days, fix_period_length
from config import time_id_max, user_id_max


def generate_w2v_data(x_data):
    data_step = user_id_max // 100
    w2v_data = np.empty([user_id_max], dtype=object)
    for user_idx in range(user_id_max):
        if (user_idx % data_step) == 0:  # 数据清洗的进度
            print(f"第 {user_idx} 条数据-->", end=';')
        user_list = w2v_data[user_idx] = []
        for day_list in x_data[user_idx]:
            if day_list is not None:
                for creative_id in day_list:
                    user_list.extend(chr(creative_id))
    print(f"第 {user_id_max} 条数据-->数据清洗完成。")
    return w2v_data


# ----------------------------------------------------------------------
def generate_fix_data(x_data):
    data_step = user_id_max // 100
    model_data = np.zeros([user_id_max, math.ceil(time_id_max / fix_period_days) * fix_period_length], dtype=object)
    for user_idx in range(user_id_max):
        if (user_idx % data_step) == 0:  # 数据清洗的进度
            print(f"第 {user_idx} 条数据-->", end=';')
        for day_idx, day_list in enumerate(x_data[user_idx]):
            if day_list is not None:
                data_offset = day_idx * fix_period_length
                for data_idx, data_value in enumerate(day_list):
                    if data_idx < fix_period_length:
                        model_data[user_idx, data_offset + data_idx] = day_list[data_idx]
    print(f"第 {user_id_max} 条数据-->数据清洗完成。")
    return model_data


def generate_day_sequence_data(x_data):
    data_step = user_id_max // 100
    sequence_data = np.zeros([user_id_max, time_id_max, 3, embedding_size], dtype=np.float16)
    from src.data.load_data import load_word2vec_weights
    embedding_weights = load_word2vec_weights()
    for user_idx in range(user_id_max):
        if (user_idx % data_step) == 0:  # 数据清洗的进度
            print(f"第 {user_idx} 条数据-->", end=';')

        for day_idx, day_list in enumerate(x_data[user_idx]):
            if day_list is not None:
                day_data = np.empty([len(day_list), embedding_size])
                for data_idx, data_value in enumerate(day_list):
                    day_data[data_idx] = embedding_weights[data_value]

                sequence_data[user_idx, day_idx, 0] = day_data.min(axis=0)
                sequence_data[user_idx, day_idx, 1] = day_data.max(axis=0)
                sequence_data[user_idx, day_idx, 2] = day_data.mean(axis=0)

    print(f"第 {user_id_max} 条数据-->数据清洗完成。")
    return sequence_data


# ----------------------------------------------------------------------
def generate_day_list_data(X_csv, y_csv):
    data_length = y_csv.shape[0]
    data_step = data_length // 100
    print(f"数据生成中（共 {data_length} 条数据)：", end='')
    X_creative_id = np.empty([user_id_max, time_id_max], dtype=object)
    creative_id_list = X_creative_id[0, 0]
    y_gender = np.empty([user_id_max], dtype=int)
    prev_user_id = prev_time_id = -1
    for i, row_data in enumerate(X_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print(f"第 {i} 条数据-->", end=';')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]
        time_id = row_data[2]
        click_times = row_data[3]

        if user_id > prev_user_id:
            y_gender[user_id] = y_csv[i]
            prev_user_id = user_id
            prev_time_id = -1

        if time_id > prev_time_id:
            prev_time_id = time_id
            creative_id_list = X_creative_id[user_id, time_id] = []

        for _ in range(click_times):
            creative_id_list.append(creative_id)
    print(f"第 {data_length} 条数据-->数据清洗完成。")
    return np.array(X_creative_id), np.array(y_gender)

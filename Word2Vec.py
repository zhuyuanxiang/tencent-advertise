# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   Tencent-Advertise
@File       :   Word2Vec.py
@Version    :   v0.1
@Time       :   2020-06-26 18:54
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""

import math
import os
import random
import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import winsound

from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec


# =====================================================
# 专门用于 Word2Vec 训练使用的数据集
def data_word2vec():
    # 清洗数据需要的变量
    user_id_num = 900000  # 用户数
    creative_id_max = 2481135 - 1  # 最大的素材编号 = 素材的总数量 - 1，这个编号已经修正了数据库与Python索引的区别
    click_times_max = 152  # 所有素材中最大的点击次数
    time_id_max = 91
    creative_id_step_size = 128000
    creative_id_window = creative_id_step_size * 5
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    # 载入数据需要的变量
    file_name = './data/train_data_all_min_complete_v.csv'
    field_list = [  # 输入数据处理：选择需要的列
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
    ]

    print('-' * 5 + "   加载数据集   " + '-' * 5)
    # 「CSV」文件字段名称
    df = pd.read_csv(file_name, dtype = int)
    # --------------------------------------------------
    # 没有在数据库中处理索引，是因为尽量不在数据库中修正原始数据，除非是不得不变更的数据，这样子业务逻辑清楚
    # user_id_inc:      X_csv[:,0]
    # creative_id_inc:  X_csv[:,1]
    # 索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的，因此字段的偏移量为 -1
    x_csv = df[field_list].values - 1
    # 'creative_id_inc' 字段的偏移量为 3，是因为需要保留 {0, 1, 2}：0 表示 “padding”（填充），1 表示 “unknown”（未知词），2 表示 “start”（序列开始）
    x_csv[:, 1] = x_csv[:, 1] + 3
    print("数据加载完成。")

    # --------------------------------------------------
    print('-' * 5 + "   清洗数据集   " + '-' * 5)
    data_length = x_csv.shape[0]
    data_step = data_length // 100  # 标识数据清洗进度的步长
    print("数据生成中（共 {0} 条数据)：".format(data_length), end = '')
    x_doc = np.zeros([user_id_num], dtype = object)

    prev_user_id = -1  # -1 不在数据序列中
    prev_time_id = -1  # 0 表示第 1 天
    for i, row_data in enumerate(x_csv):
        if (i % data_step) == 0:  # 数据清洗的进度
            print("第 {0} 条数据-->".format(i), end = ';')
            pass
        user_id = row_data[0]
        time_id = row_data[2]

        # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
        if user_id > prev_user_id:
            prev_user_id = user_id
            prev_time_id = 0
            x_doc[user_id] = [2]
            pass

        if (time_id - 1) > prev_time_id:  # 时间间隔超过 1 天的就插入 0
            x_doc[user_id].append(0)

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

        x_doc[user_id].append(creative_id)
        prev_time_id = time_id
    print("\n数据清洗完成！")
    return x_doc.tolist()


path = get_tmpfile('save_model/word2vec.model')
words_lists = data_word2vec()
print(words_lists[10])
for i, one_list in enumerate(words_lists):
    for j, number in enumerate(one_list):
        words_lists[i][j] = str(number)

model = Word2Vec(words_lists, size = 32, window = 5, min_count = 1, workers = 8)
model.save('save_model/word2vec.model')

if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

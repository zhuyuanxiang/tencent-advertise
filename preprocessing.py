# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
--------------------------------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   preprocessing.py
@Version    :   v0.1
@Time       :   2020-06-07 15:41
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   数据预处理模块
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
from tensorflow import keras

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ >= "1.18.1"
# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3,
                    suppress=True,
                    threshold=np.inf,
                    linewidth=200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)


# ----------------------------------------------------------------------
def load_original_data(file_name, field_list, label_list):
    '''
    从 csv 文件中载入原始数据
    :param file_name: 载入数据的文件名
    :return: x_csv, y_csv
    '''
    print('-' * 5 + "   加载数据集   " + '-' * 5)
    # 「CSV」文件字段名称
    df = pd.read_csv(file_name, dtype=int)
    # --------------------------------------------------
    # 没有在数据库中处理索引，是因为尽量不在数据库中修正原始数据，除非是不得不变更的数据，这样子业务逻辑清楚
    # user_id_inc:      X_csv[:,0]
    # creative_id_inc:  X_csv[:,1]
    # time_id:          X_csv[:,2]
    # click_times:      X_csv[:,3]
    # 索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的，因此字段的偏移量为 -1
    x_csv = df[field_list].values - 1
    # 'creative_id_inc' 字段的偏移量为 3，是因为需要保留 {0, 1, 2}：0 表示 “padding”（填充），1 表示 “unknown”（未知词），2 表示 “start”（序列开始）
    x_csv[:, 1] = x_csv[:, 1] + 3
    x_csv[:, 3] = x_csv[:, 3] + 1  # 数据是值，不需要减 1
    # --------------------------------------------------
    # 目标数据处理：目标字段的偏移量是 -1，是因为索引在数据库中是从 1 开始的，在 Python 中是从 0 开始的
    # 既可以加载 'age'，也可以加载 'gender'
    y_csv = df[label_list].values - 1
    print("数据加载完成。")
    output_example_data(x_csv, y_csv, '加载数据')
    return x_csv, y_csv


# ----------------------------------------------------------------------
def output_example_data(X, y, data_type='原始数据'):
    print(data_type + ":(X[0], y[0]) =", X[0], y[0])
    print(data_type + ":(X[30], y[30]) =", X[30], y[30])
    print(data_type + ":(X[600], y[600]) =", X[600], y[600])
    if len(y) > 8999:
        print(data_type + "：(X[8999], y[8999]) =", X[8999], y[8999])
    if len(y) > 119999:
        print(data_type + "：(X[119999], y[119999]) =", X[119999], y[119999])
    if len(y) > 224999:
        print(data_type + "：(X[224999], y[224999]) =", X[224999], y[224999])
    if len(y) > 674999:
        print(data_type + "：(X[674999], y[674999]) =", X[674999], y[674999])
    if len(y) > 899999:
        print(data_type + "：(X[899999], y[899999]) =", X[899999], y[899999])
        pass
    pass


# ----------------------------------------------------------------------
def data_no_sequence(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    # -1 不在数据序列中
    tmp_user_id = -1
    tmp_creative_id = -1
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        user_id = row_data[0]
        creative_id = row_data[1]

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                tmp_user_id = user_id
                # 新建用户序列时，数据序列用 2 表示用户序列的开始，标签序列更新为用户的标签
                X_doc[user_id] = [2]
                y_doc[user_id] = y_data[i]
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].append(creative_id)
            tmp_creative_id = creative_id
            pass
        pass
    pass
    print("\n数据清洗完成！")
    print("清洗数据(X_doc[0], y_doc[0]) =", X_doc[0], y_doc[0])
    print("清洗数据(X_doc[30], y_doc[30]) =", X_doc[30], y_doc[30])
    print("清洗数据(X_doc[600], y_doc[600]) =", X_doc[600], y_doc[600])
    print("清洗数据(X_doc[9000], y_doc[9000]) =", X_doc[9000], y_doc[9000])
    return X_doc, y_doc


# ----------------------------------------------------------------------
# 生成的用户序列数据：0 表示这天没有访问素材；1 表示这个素材不在词典中
def data_sequence_no_start(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        creative_id = row_data[1]
        user_id = row_data[2] - 1  # 索引从 0 开始

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                tmp_user_id = user_id
                X_doc[user_id] = []
                y_doc[user_id] = y_data[i]
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].append(creative_id)
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
# 生成的用户序列数据：2 表示用户访问序列的开始；0 表示这天没有访问素材；1 表示这个素材不在词典中
# 序列中重复的数据是因为某个素材访问好几次；最后的0是填充数据
# 生成每个用户的数据序列，按天数排序
# data_sequence()
#       点击次数超过 1 次的也只有一条数据，没有数据的天就跳过(不填充0)
# data_sequence_with_interval()
#       点击次数超过 1 次的也只有一条数据，没有数据的天数就插入一个0，不管差几天
# data_sequence_reverse()
#       点击次数超过 1 次的也只有一条数据，没有数据的天就跳过(不填充0)，头和尾各增加3天数据
# data_sequence_reverse_with_interval()
#       点击次数超过 1 次的也只有一条数据，没有数据的天数就插入一个0，头和尾各增加5天数据
#
# data_sequence_times()
#       点击次数超过 1 次的数据重复生成，没有数据的天就跳过
# data_sequence_times_with_interval()
#       点击次数超过 1 次的数据重复生成，没有数据的天数就插入一个0，不管差几天
# data_sequence_times_with_empty()
#       点击次数超过 1 次的数据重复生成，没有数据的天就置 0，差几天就填充几个 0
# data_sequence_with_fix()
#       每天固定几个数据，没有数据的天就置0，超过范围的数据就截断，点击次数超过 1 次的也只有一条数据，
# ----------------------------------------------------------------------
def data_sequence(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        creative_id = row_data[1]
        user_id = row_data[2] - 1  # 索引从 0 开始

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                tmp_user_id = user_id
                # 新建用户序列时，数据序列用 2 表示用户序列的开始，标签序列更新为用户的标签
                X_doc[user_id] = [2]
                y_doc[user_id] = y_data[i]
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].append(creative_id)
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def data_sequence_with_interval(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        time_id = row_data[0]
        creative_id = row_data[1]
        user_id = row_data[2] - 1  # 索引从 0 开始

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                tmp_user_id = user_id
                tmp_time_id = 0
                X_doc[user_id] = [2]  # 2 表示序列的开始
                # 新建用户序列时，更新用户的标签
                y_doc[user_id] = y_data[i]
                pass
            if tmp_time_id < time_id:
                # 按照时间差更新用户序列中的空缺天数，两个天数之间的空缺天数=后-次的天数-前一次的天数-1
                X_doc[user_id].append(0)
                tmp_time_id = time_id
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].append(creative_id)
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def data_sequence_reverse_with_interval(X_data, y_data, user_id_num,
                                        creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        creative_id = row_data[1]
        user_id = row_data[2] - 1  # 索引从 0 开始

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                # 新建用户序列前，将旧的用户序列前面增加3天数据，后面增加3天数据
                # 最后一个用户序列需要单独更新，因为不再有新的用户序列激活这个更新操作
                extend_user_id_sequence(X_doc, tmp_user_id)
                tmp_user_id = user_id
                # 新建用户序列时，数据序列用 2 表示用户序列的开始，标签序列更新为用户的标签
                X_doc[user_id] = [2]  #
                y_doc[user_id] = y_data[i]
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].append(creative_id)
            pass
        pass
    extend_user_id_sequence(X_doc, tmp_user_id)
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def extend_user_id_sequence(X_doc, extend_user_id, extend_len):
    pre_user_id_list = X_doc[extend_user_id].copy()
    user_id_len = len(pre_user_id_list)
    if user_id_len < extend_len:
        extend_len = user_id_len - 1  # 减 1 是要减去序列开始的那个编号「1」
    X_doc[extend_user_id][1:1] = pre_user_id_list[user_id_len - extend_len:]
    X_doc[extend_user_id].extend(pre_user_id_list[1:1 + extend_len])


# ----------------------------------------------------------------------
def data_sequence_times(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        time_id = row_data[0]
        creative_id = row_data[1]
        user_id = row_data[2] - 1  # 索引从 0 开始
        click_times = row_data[3]

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                tmp_user_id = user_id
                tmp_time_id = 0
                X_doc[user_id] = [2]  # 2 表示序列的开始
                # 新建用户序列时，更新用户的标签
                y_doc[user_id] = y_data[i]
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].extend([creative_id for _ in range(click_times)
                                   ])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def data_sequence_times_with_interval(X_data, y_data, user_id_num,
                                      creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        time_id = row_data[0]
        creative_id = row_data[1]
        user_id = row_data[2] - 1  # 索引从 0 开始
        click_times = row_data[3]

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                tmp_user_id = user_id
                tmp_time_id = 0
                X_doc[user_id] = [2]  # 2 表示序列的开始
                # 新建用户序列时，更新用户的标签
                y_doc[user_id] = y_data[i]
                pass
            if tmp_time_id < time_id:
                # 按照时间差更新用户序列中的空缺天数，两个天数之间的空缺天数=后-次的天数-前一次的天数-1
                X_doc[user_id].append(0)
                tmp_time_id = time_id
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].extend([creative_id for _ in range(click_times)
                                   ])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def data_sequence_times_with_empty(X_data, y_data, user_id_num,
                                   creative_id_num):
    # 生成每个用户的数据序列，按天数排序，没有数据的天就置0
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_num], dtype=object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    # 生成的用户序列数据：2 表示用户访问序列的开始；0 表示这天没有访问素材；1 表示这个素材不在词典中
    # 序列中重复的数据是因为某个素材访问好几次；最后的0是填充数据
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end='')
            pass
        time_id = row_data[0]
        creative_id = row_data[1]
        user_id = row_data[2] - 1  # 索引从 0 开始
        click_times = row_data[3]

        # user_id 是否属于关注的用户范围，访问素材数量过低的用户容易成为噪声
        if user_id < user_id_num:
            # 原始数据的 user_id 已经排序了，因此出现新的 user_id 时，就新建一个用户序列
            if user_id != tmp_user_id:
                tmp_user_id = user_id
                tmp_time_id = 0
                X_doc[user_id] = [2]  # 2 表示序列的开始
                # 新建用户序列时，更新用户的标签
                y_doc[user_id] = y_data[i]
                pass
            if tmp_time_id < time_id:
                # 按照时间差更新用户序列中的空缺天数，两个天数之间的空缺天数=后-次的天数-前一次的天数-1
                X_doc[user_id].extend(
                    [0 for _ in range(time_id - tmp_time_id - 1)])
                tmp_time_id = time_id
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].extend([creative_id for _ in range(click_times)
                                   ])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def generate_fix_data(X_csv, y_csv, train_field_num):
    print('-' * 5 + "   清洗数据集   " + '-' * 5)
    print("数据生成中（共 {0} 条数据)：".format(30000000), end='')
    # 初始化 X_doc 为空的列表 : X_doc[:,0]: creative_id, X_doc[:,1]: click_times
    # 初始化 y_doc 为空的列表 : y_doc[:,0]: age, X_doc[:,1]: gender
    X_doc = np.zeros([
        user_id_num, train_field_num,
        time_id_max * period_length // period_days
    ],
                     dtype=object)
    y_doc = np.zeros([user_id_num, 2], dtype=int)
    data_step = X_csv.shape[0] // 100  # 标识数据清洗进度的步长
    prev_user_id = -1
    prev_time_id = 0
    period_index = 0
    for i, row_data in enumerate(X_csv):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print("第 {0} 条数据-->".format(i), end=';')
            pass
        user_id = row_data[0]
        time_id = row_data[2]
        if user_id >= user_id_num:
            break
        y_doc[user_id, 0] = y_csv[i, 0]
        y_doc[user_id, 1] = y_csv[i, 1]
        # 整理过的数据已经按照 user_id 的顺序编号，当 user_id 变化时，就代表前一个用户的数据已经清洗完成
        if user_id > prev_user_id:
            # 重置临时变量
            prev_user_id = user_id
            prev_time_id = 0
            period_index = 0
            pass

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

        X_doc[user_id, 0, (time_id // period_days) * period_length +
              period_index] = creative_id
        X_doc[user_id, 1, (time_id // period_days) * period_length +
              period_index] = click_times
        period_index = period_index + 1
        pass
    print("\n数据清洗完成！")
    output_example_data(X_doc, y_doc, data_type='清洗数据')
    return X_doc, y_doc


# ----------------------------------------------------------------------
def split_data(x_data, y_data, label_name):
    print('-' * 5 + "   拆分{0}数据集   " + label_name + '-' * 5)
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        random_state=seed,
                                                        stratify=y_data)
    data_type = "训练数据集"
    print('-' * 5 + "   {0}：{1}   ".format(data_type, len(y_train)) + '-' * 5)
    output_example_data(x_train, y_train, data_type)
    np.save('save_data/x_train_fix_' + label_name, x_train)
    np.save('save_data/y_train_fix_' + label_name, y_train)
    data_type = "测试数据集"
    print('-' * 5 + "   {0}：{1}   ".format(data_type, len(y_train)) + '-' * 5)
    output_example_data(x_test, y_test, data_type)
    np.save('save_data/x_test_fix_' + label_name, x_test)
    np.save('save_data/y_test_fix_' + label_name, y_test)
    return x_train, x_test, y_train, y_test


# ----------------------------------------------------------------------
def data_sequence_with_fix():
    # 载入数据需要的变量
    file_name = './data/train_data_all_min_complete_v.csv'
    field_list = [  # 输入数据处理：选择需要的列
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]
    field_num = 4
    label_list = ['age', 'gender']
    train_field_num = field_num - 2  # filed_list 去除 user_id, time_id

    # --------------------------------------------------
    # 加载数据
    x_csv, y_csv = load_original_data(file_name, field_list, label_list)

    # --------------------------------------------------
    # 清洗数据集，生成所需要的数据
    x_data, y_data = generate_fix_data(x_csv, y_csv, train_field_num)

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    x_train_age, x_test_age, y_train_age, y_test_age = split_data(
        x_data, y_data[:, 0], label_list[0])  # 拆分「age」多分类数据

    x_train_gender, x_test_gender, y_train_gender, y_test_gender = split_data(
        x_data, y_data[:, 1], label_list[1])  # 拆分「gender」二分类数据
    pass


# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 清洗数据需要的变量
    user_id_num = 900000  # 用户数
    creative_id_max = 2481135 - 1  # 最大的素材编号 = 素材的总数量 - 1，这个编号已经修正了数据库与Python索引的区别
    click_times_max = 152  # 所有素材中最大的点击次数
    time_id_max = 91
    period_length = 21
    period_days = 7
    creative_id_step_size = 128000
    creative_id_window = creative_id_step_size * 5
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    data_sequence_with_fix()
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

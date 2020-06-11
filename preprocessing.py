# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
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
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)


# ----------------------------------------------------------------------
def load_data(file_name, label_name = 'gender'):
    '''
    从 csv 文件中载入原始数据
    :param file_name: 载入数据的文件名
    :param label_name: 返回的标签的名称
    可以定义为 `age`：年龄 或者 `gender`：性别
    :return:
    '''
    print("* 加载数据集...")
    # 「CSV」文件字段名称
    # "time_id","user_id_inc","user_id","creative_id_inc","creative_id","click_times","age","gender"
    df = pd.read_csv(file_name)
    # 选择需要的列作为输入数据
    X_data = df[["time_id", "creative_id_inc", "user_id_inc", "click_times"]].values
    # 索引在数据库中是从 1 开始的，加上 2 ，是因为 Python 的索引是从 0 开始的
    # 并且需要保留 {0, 1, 2} 三个数：0 表示 “padding”（填充），1 表示 “unknown”（未知词），2 表示 “start”（用户开始）
    X_data[:, 1] = X_data[:, 1] + 2
    y_data = df[label_name].values - 1  # 目标数据
    print("数据加载完成。")
    return X_data, y_data


# ----------------------------------------------------------------------
# 生成的用户序列数据：0 表示这天没有访问素材；1 表示这个素材不在词典中
def data_sequence_no_start(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
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
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
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
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
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
def data_sequence_reverse_with_interval(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
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
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
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
            X_doc[user_id].extend([creative_id for _ in range(click_times)])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def data_sequence_times_with_interval(X_data, y_data, user_id_num, creative_id_num):
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
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
            X_doc[user_id].extend([creative_id for _ in range(click_times)])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


# ----------------------------------------------------------------------
def data_sequence_times_with_empty(X_data, y_data, user_id_num, creative_id_num):
    # 生成每个用户的数据序列，按天数排序，没有数据的天就置0
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    # 生成的用户序列数据：2 表示用户访问序列的开始；0 表示这天没有访问素材；1 表示这个素材不在词典中
    # 序列中重复的数据是因为某个素材访问好几次；最后的0是填充数据
    for i, row_data in enumerate(X_data):
        # 数据清洗的进度
        if (i % data_step) == 0:
            print(".", end = '')
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
                X_doc[user_id].extend([0 for _ in range(time_id - tmp_time_id - 1)])
                tmp_time_id = time_id
                pass
            # 超过词典大小的素材标注为 1，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 1
                pass
            X_doc[user_id].extend([creative_id for _ in range(click_times)])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


def data_sequence_times_with_empty():
    pass


def data_sequence_with_fix():
    pass


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

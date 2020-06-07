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
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ >= "1.18.1"


# ----------------------------------------------------------------------
def data_sequence():
    # 生成每个用户的数据序列，按天数排序，没有数据的天就跳过
    pass


def data_sequence_times(X_data, y_data, user_id_num, creative_id_num):
    # 生成每个用户的数据序列，按天数排序，点击次数超过 1 次的数据重复生成，没有数据的天就跳过
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    # 生成的用户序列数据：1 表示用户访问序列的开始；0 表示这天没有访问素材；2 表示这个素材不在词典中
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
                X_doc[user_id] = [1]  # 1 表示序列的开始
                # 新建用户序列时，更新用户的标签
                y_doc[user_id] = y_data[i]
                pass
            # 超过词典大小的素材标注为 2，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 2
                pass
            X_doc[user_id].extend([creative_id for _ in range(click_times)])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    # padding: 字符串，'pre' 或 'post' ，在序列的前端补齐还是在后端补齐。
    print("\n数据清洗完成！")
    return X_doc, y_doc


def data_sequence_with_empty(X_data, y_data, user_id_num, creative_id_num):
    # 生成每个用户的数据序列，按天数排序，没有数据的天就置0
    print("数据清洗中：", end = '')
    X_doc = np.zeros([user_id_num], dtype = object)
    y_doc = np.zeros([user_id_num])
    tmp_user_id = -1  # -1 表示 id 不在数据序列中
    tmp_time_id = 0
    data_step = X_data.shape[0] // 100  # 标识数据清洗进度的步长
    # 生成的用户序列数据：1 表示用户访问序列的开始；0 表示这天没有访问素材；2 表示这个素材不在词典中
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
                X_doc[user_id] = [1]  # 1 表示序列的开始
                # 新建用户序列时，更新用户的标签
                y_doc[user_id] = y_data[i]
                pass
            if tmp_time_id < time_id:
                # 按照时间差更新用户序列中的空缺天数，两个天数之间的空缺天数=后-次的天数-前一次的天数-1
                X_doc[user_id].extend([0 for _ in range(time_id - tmp_time_id - 1)])
                tmp_time_id = time_id
                pass
            # 超过词典大小的素材标注为 2，即「未知」
            if creative_id >= creative_id_num:
                creative_id = 2
                pass
            X_doc[user_id].extend([creative_id for _ in range(click_times)])  # 按照点击次数更新用户序列
            pass
        pass
    pass
    print("\n数据清洗完成！")
    return X_doc, y_doc


def data_sequence_times_with_empty():
    # 生成每个用户的数据序列，按天数排序，点击次数超过 1 次的数据重复生成，没有数据的天就置0
    pass


def data_sequence_with_fix():
    # 生成每个用户的数据序列，按天数排序，每天固定几个数据，没有数据的天就置0
    pass


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

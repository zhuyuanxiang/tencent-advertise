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
from sklearn.model_selection import train_test_split

import config
import numpy as np

from config import creative_id_step_size, seed
from code.无用代码.generate_data import generate_word2vec_data_with_interval, generate_word2vec_data_no_interval, \
    generate_fix_data, generate_no_time_data
from code.data.load_data import load_original_data
from code.无用代码.load_data import load_word2vec_file
from code.无用代码.save_data import save_data_set, save_word2vec_data
from code.data.show_data import show_word2vec_data


# ----------------------------------------------------------------------
# 生成的用户序列数据：2 表示用户访问序列的开始；0 表示这天没有访问素材；1 表示这个素材不在词典中
# 序列中重复的数据是因为某个素材访问好几次
# 生成的数据就是原始数据，对于序列数据还需要在未来按照固定长度填充
# 生成无 time_id 标志的数据
# data_no_time()
#       每个用户的数据按照访问顺序排列，不考虑访问的时间点，重复的数据也不放入用户的序列中
#       使用最大池化进行学习
# 生成每个用户的数据序列，按天数排序
# data_sequence()
#       点击次数超过 1 次的也只有一条数据，没有数据的天就跳过(不填充0)
# data_sequence_with_interval()
#       点击次数超过 1 次的也只有一条数据，没有数据的天数就插入一个0，不管差几天
# data_sequence_reverse()
#       点击次数超过 1 次的也只有一条数据，没有数据的天就跳过(不填充0)，头和尾各增加3天数据
# data_sequence_reverse_with_interval()
#       点击次数超过 1 次的也只有一条数据，没有数据的天数就插入一个0，头和尾各增加5天数据
# data_sequence_times()
#       点击次数超过 1 次的数据重复生成，没有数据的天就跳过
# data_sequence_times_with_interval()
#       点击次数超过 1 次的数据重复生成，没有数据的天数就插入一个0，不管差几天
# data_sequence_times_with_empty()
#       点击次数超过 1 次的数据重复生成，没有数据的天就置 0，差几天就填充几个 0
# 生成固定周期长度的数据
#       每个周期固定几个数据，没有数据的天就置 0，超过范围的数据就截断，不足的以 0 补齐，
#       点击次数超过 1 次的也只有一条数据，不同 time_id 重复访问的数据会出现多条数据
#       使用卷积神经网络进行特征提取
# data_fix()
#       生成多个周期定义的固定长度数据，因为内存不足因此无法使用
# data_fix_day()
#       生成每天固定长度的数据
# data_fix_three_days()
#       生成每三天固定长度的数据
# data_fix_week()
#       生成每周固定长度的数据
# ----------------------------------------------------------------------
from tools import beep_end


def data_no_time():
    # 载入数据需要的变量
    file_name = './data/train_data_all_min_complete_v.csv'
    field_list = [  # 输入数据处理：选择需要的列
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]

    # --------------------------------------------------
    # 加载数据
    label_list = ['age', 'gender']
    x_csv, y_csv = load_original_data()

    # --------------------------------------------------
    # 清洗数据集，生成不重复的数据，用于 MaxPooling()训练
    file_suffix = 'no_time_no_repeat'
    x_data, y_data = generate_no_time_data(x_csv, y_csv, len(field_list) - 2, len(label_list), repeat_creative_id=False)

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # 拆分「gender」二分类数据
    label_name = label_list[1]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 1], label_name)
    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender, x_data, y_data

    # --------------------------------------------------
    # 清洗数据集，生成可重复的数据，用于 Word2Vec 训练
    file_suffix = 'no_time_with_repeat'
    x_data, y_data = generate_no_time_data(x_csv, y_csv, len(field_list) - 2, len(label_list), repeat_creative_id=True)

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # 拆分「gender」二分类数据
    label_name = label_list[1]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 1], label_name)
    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender, x_data, y_data


# ----------------------------------------------------------------------
def data_sequence(X_data, y_data, user_id_max, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_max], dtype=object)
    y_doc = np.zeros([user_id_max])
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
        if user_id < user_id_max:
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
# 生成的用户序列数据：0 表示这天没有访问素材；1 表示这个素材不在词典中
def data_sequence_no_start(X_data, y_data, user_id_max, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_max], dtype=object)
    y_doc = np.zeros([user_id_max])
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
        if user_id < user_id_max:
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
# 程序虽然简单，但是机器内存不够，根本无法执行
def data_fix():
    # 载入数据需要的变量
    file_name = './data/train_data_all_min_complete_v.csv'
    field_list = [  # 输入数据处理：选择需要的列
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]
    label_list = ['age', 'gender']

    # --------------------------------------------------
    # 加载数据
    x_csv, y_csv = load_original_data()

    # ==================================================
    # 清洗数据集，生成所需要的数据 ( 每人每天访问素材的数量 )
    period_length = 7
    period_days = 1
    x_data, y_data = generate_fix_data(x_csv, y_csv)

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    file_suffix = 'fix_day'
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # 拆分「gender」二分类数据
    label_name = label_list[1]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 1], label_name)
    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender

    del x_data, y_data

    # ==================================================
    # 清洗数据集，生成所需要的数据 ( 每人每天访问素材的数量 )
    period_length = 15
    period_days = 3
    x_data, y_data = generate_fix_data(x_csv, y_csv)

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    file_suffix = 'fix_three_days'
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # 拆分「gender」二分类数据
    label_name = label_list[1]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 1], label_name)
    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender

    del x_data, y_data

    # ==================================================
    # 清洗数据集，生成所需要的数据 ( 每人每天访问素材的数量 )
    period_length = 21
    period_days = 7
    x_data, y_data = generate_fix_data(x_csv, y_csv)

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    file_suffix = 'fix_week'
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # 拆分「gender」二分类数据
    label_name = label_list[1]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 1], label_name)
    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender

    del x_data, y_data

    pass


# ----------------------------------------------------------------------
# 数据量太大，使用更小的粒度操作数据
def data_fix_day():
    # 载入数据需要的变量
    file_name = './data/train_data_all_min_complete_v.csv'
    field_list = [  # 输入数据处理：选择需要的列
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]
    file_suffix = 'fix_day'
    period_length = 7
    period_days = 1

    # --------------------------------------------------
    # 加载数据
    label_list = ['age']
    x_csv, y_csv = load_original_data()

    # --------------------------------------------------
    # 清洗数据集，生成所需要的数据
    x_data, y_data = generate_fix_data(x_csv, y_csv)
    del x_csv, y_csv

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    del x_data, y_data

    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # --------------------------------------------------
    # 加载数据
    label_list = ['gender']
    x_csv, y_csv = load_original_data()

    # --------------------------------------------------
    # 清洗数据集，生成所需要的数据
    x_data, y_data = generate_fix_data(x_csv, y_csv)
    del x_csv, y_csv

    # 拆分「gender」二分类数据
    label_name = label_list[0]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 0], label_name)
    del x_data, y_data

    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender
    pass


# ----------------------------------------------------------------------
def data_fix_three_days():
    # 载入数据需要的变量
    file_name = './data/train_data_all_min_complete_v.csv'
    field_list = [  # 输入数据处理：选择需要的列
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]
    label_list = ['age', 'gender']
    file_suffix = 'fix_three_days'
    period_length = 15
    period_days = 3

    # --------------------------------------------------
    # 加载数据
    x_csv, y_csv = load_original_data()

    # --------------------------------------------------
    # 清洗数据集，生成所需要的数据
    x_data, y_data = generate_fix_data(x_csv, y_csv)
    del x_csv, y_csv

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # 拆分「gender」二分类数据
    label_name = label_list[1]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 1], label_name)
    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender
    del x_data, y_data
    pass


# ----------------------------------------------------------------------
def data_fix_week():
    # 载入数据需要的变量
    file_name = './data/train_data_all_min_complete_v.csv'
    field_list = [  # 输入数据处理：选择需要的列
        "user_id_inc",  # 0
        "creative_id_inc",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]
    label_list = ['age', 'gender']
    file_suffix = 'fix_week'
    period_length = 21
    period_days = 7

    # --------------------------------------------------
    # 加载数据
    x_csv, y_csv = load_original_data()

    # --------------------------------------------------
    # 清洗数据集，生成所需要的数据
    x_data, y_data = generate_fix_data(x_csv, y_csv)
    del x_csv, y_csv

    # --------------------------------------------------
    # 拆分数据集，按 3:1 分成 训练数据集 和 测试数据集
    # 拆分「age」多分类数据
    label_name = label_list[0]
    x_train_age, y_train_age, x_test_age, y_test_age = split_data(x_data, y_data[:, 0], label_name)
    save_data_set(x_train_age, y_train_age, x_test_age, y_test_age)
    del x_train_age, y_train_age, x_test_age, y_test_age

    # 拆分「gender」二分类数据
    label_name = label_list[1]
    x_train_gender, y_train_gender, x_test_gender, y_test_gender = split_data(x_data, y_data[:, 1], label_name)
    save_data_set(x_train_gender, y_train_gender, x_test_gender, y_test_gender)
    del x_train_gender, y_train_gender, x_test_gender, y_test_gender
    del x_data, y_data
    pass


# ----------------------------------------------------------------------
def data_sequence_with_interval(X_data, y_data, user_id_max, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_max], dtype=object)
    y_doc = np.zeros([user_id_max])
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
        if user_id < user_id_max:
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
def data_sequence_reverse_with_interval(X_data, y_data, user_id_max, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_max], dtype=object)
    y_doc = np.zeros([user_id_max])
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
        if user_id < user_id_max:
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
def data_sequence_times(X_data, y_data, user_id_max, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_max], dtype=object)
    y_doc = np.zeros([user_id_max])
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
        if user_id < user_id_max:
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
def data_sequence_times_with_interval(X_data, y_data, user_id_max, creative_id_num):
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_max], dtype=object)
    y_doc = np.zeros([user_id_max])
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
        if user_id < user_id_max:
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
def data_sequence_times_with_empty(X_data, y_data, user_id_max, creative_id_num):
    # 生成每个用户的数据序列，按天数排序，没有数据的天就置0
    print("数据清洗中：", end='')
    X_doc = np.zeros([user_id_max], dtype=object)
    y_doc = np.zeros([user_id_max])
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
        if user_id < user_id_max:
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


# =====================================================
# 生成用于 Word2Vec 训练使用的数据集
def data_word2vec():
    x_csv = load_word2vec_file('', [])

    config.creative_id_window = creative_id_step_size * 5
    config.creative_id_begin = creative_id_step_size * 0
    config.creative_id_end = config.creative_id_begin + config.creative_id_window
    x_creative_id = generate_word2vec_data_with_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, config.creative_id_window, '')
    del x_creative_id
    x_creative_id = generate_word2vec_data_no_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, config.creative_id_window, '')
    del x_creative_id

    config.creative_id_window = creative_id_step_size * 8
    config.creative_id_begin = creative_id_step_size * 0
    config.creative_id_end = config.creative_id_begin + config.creative_id_window
    x_creative_id = generate_word2vec_data_with_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, config.creative_id_window, '')
    del x_creative_id
    x_creative_id = generate_word2vec_data_no_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, config.creative_id_window, '')
    del x_creative_id

    print("\n数据清洗完成！")


# ----------------------------------------------------------------------
def extend_user_id_sequence(X_doc, extend_user_id, extend_len):
    pre_user_id_list = X_doc[extend_user_id].copy()
    user_id_len = len(pre_user_id_list)
    if user_id_len < extend_len:
        extend_len = user_id_len - 1  # 减 1 是要减去序列开始的那个编号「1」
    X_doc[extend_user_id][1:1] = pre_user_id_list[user_id_len - extend_len:]
    X_doc[extend_user_id].extend(pre_user_id_list[1:1 + extend_len])


# ----------------------------------------------------------------------
if __name__ == '__main__':
    # data_no_time()
    # data_fix_day()  # 固定每人每天访问素材的数量
    # data_fix_three_days()  # 固定每人每三天访问素材的数量
    # data_fix_week()  # 固定每人每周访问素材的数量
    data_word2vec()
    # 运行结束的提醒
    beep_end()


def split_data(x_data, y_data, label_name):
    print('-' * 5 + "   拆分{0}数据集   ".format(label_name) + '-' * 5)
    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=seed, stratify=y_data)
    return x_train, y_train, x_test, y_test

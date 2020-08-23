# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   export_data.py
@Version    :   v0.1
@Time       :   2020-08-19 12:11
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   导出数据模块
@理解：
"""
from sklearn.model_selection import train_test_split

import config
from config import label_name
from src.data.save_data import save_model_data
from src.无用代码.generate_data import generate_balance_data
from tools import show_title


# ----------------------------------------------------------------------
def export_val_balance(x_train_val, y_train_val):
    show_title(f"对类别 ：{label_name}实施平衡{config.train_val_data_type}")
    x_train_val_balance, y_train_val_balance = generate_balance_data(x_train_val, y_train_val)
    save_model_data(x_train_val_balance, config.data_file_path + 'x_train_val_balance', '平衡拆分验证的训练数据集')
    save_model_data(y_train_val_balance, config.data_file_path + 'y_train_val_balance', '平衡拆分验证的训练数据集')


def export_train_balance(x_train, y_train):
    show_title(f"对类别 ：{label_name}实施平衡{config.train_data_type}")
    x_train_balance, y_train_balance = generate_balance_data(x_train, y_train)
    save_model_data(x_train_balance, config.data_file_path + config.x_train_balance_file_name, '平衡的训练数据集')
    save_model_data(y_train_balance, config.data_file_path + config.y_train_balance_file_name, '平衡的训练数据集')


def export_val_data(x_train, y_train):
    show_title("拆分训练数据集和验证数据集")
    x_train_val, x_val, y_train_val, y_val = train_test_split(
            x_train, y_train, random_state=config.seed, stratify=y_train)
    save_model_data(x_train_val, config.data_file_path + config.x_train_val_file_name, '无验证训练数据集')
    save_model_data(y_train_val, config.data_file_path + config.y_train_val_file_name, '无验证训练数据集')
    save_model_data(x_val, config.data_file_path + config.x_val_file_name, '验证数据集')
    save_model_data(y_val, config.data_file_path + config.y_val_file_name, '验证数据集')
    return x_train_val, y_train_val


def export_train_test_data(x_data, y_data):
    show_title("拆分训练数据集和测试数据集")
    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=config.seed, stratify=y_data)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    save_model_data(x_train, config.data_file_path + config.x_train_file_name, '训练数据集')
    save_model_data(y_train, config.data_file_path + config.y_train_file_name, '训练数据集')
    save_model_data(x_test, config.data_file_path + config.x_test_file_name, '测试数据集')
    save_model_data(y_test, config.data_file_path + config.y_test_file_name, '测试数据集')
    return x_train, y_train


def export_generate_data(lst_data):
    from src.data.generate_data import generate_day_sequence_data
    show_title("加工数据为 91 天的序列数据，每天为6个特征(最大值、最小值、平均值)96维数据")
    x_data = generate_day_sequence_data(lst_data)
    save_model_data(x_data, config.data_file_path + config.x_data_file_name, '中间数据集')

    # show_title("加工数据为定长的数据列表")
    # from src.data.generate_data import generate_fix_data
    # x_data = generate_fix_data(lst_data)

    # show_title("加工数据为无间隔有重复的数据列表")
    # from generate_data import generate_data_no_interval_with_repeat
    # x_data, y_data, x_w2v = generate_data_no_interval_with_repeat(x_csv, y_csv)
    return x_data


def export_base_data():
    lst_data, y_data = export_day_list_data()
    export_w2v_data(lst_data)
    # from src.data.load_data import load_day_list_data
    # lst_data, y_data = load_day_list_data()
    x_data = export_generate_data(lst_data)
    return x_data, y_data


def export_day_list_data():
    from src.data.load_data import load_original_data
    show_title("加载原始数据")
    x_csv, y_csv = load_original_data()

    from src.data.generate_data import generate_day_list_data
    show_title("导出每个用户每天访问数据的不截断列表")
    lst_data, y_data = generate_day_list_data(x_csv, y_csv)
    save_model_data(lst_data, config.data_file_path + config.lst_data_file_name, '中间数据集')
    save_model_data(y_data, config.data_file_path + config.y_data_file_name, '中间数据集')
    return lst_data, y_data


def export_w2v_data(lst_data):
    show_title("导出用于Word2Vec训练的数据")
    from src.data.generate_data import generate_w2v_data
    save_model_data(generate_w2v_data(lst_data), config.data_w2v_path + 'x_w2v', 'w2v数据集')

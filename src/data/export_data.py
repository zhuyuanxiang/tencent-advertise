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

from src.base import config
from src.base.classes import ExportDataType
from src.base.config import data_file_path
from src.base.config import label_name
from src.base.tools import show_title
from src.data.save_data import save_model_data
from src.无用代码.generate_data import generate_balance_data


# ----------------------------------------------------------------------
def export_val_balance(x_train_val, y_train_val):
    show_title(f"对类别 ：{label_name}实施平衡{config.train_val_data_type}")
    x_train_val_balance, y_train_val_balance = generate_balance_data(x_train_val, y_train_val)

    from src.base.config import train_val_balance_data_type
    from src.base.config import x_train_val_balance_file_name, y_train_val_balance_file_name
    save_model_data(x_train_val_balance, data_file_path + x_train_val_balance_file_name, train_val_balance_data_type)
    save_model_data(y_train_val_balance, data_file_path + y_train_val_balance_file_name, train_val_balance_data_type)


def export_train_balance(x_train, y_train):
    show_title(f"对类别 ：{label_name}实施平衡{config.train_data_type}")
    x_train_balance, y_train_balance = generate_balance_data(x_train, y_train)

    from src.base.config import train_balance_data_type
    from src.base.config import x_train_balance_file_name, y_train_balance_file_name
    save_model_data(x_train_balance, data_file_path + x_train_balance_file_name, train_balance_data_type)
    save_model_data(y_train_balance, data_file_path + y_train_balance_file_name, train_balance_data_type)


def export_val_data(x_train, y_train):
    show_title("拆分训练数据集和验证数据集")
    x_train_val, x_val, y_train_val, y_val = train_test_split(
            x_train, y_train, random_state=config.seed, stratify=y_train)

    from src.base.config import train_val_data_type
    from src.base.config import x_train_val_file_name, y_train_val_file_name
    save_model_data(x_train_val, data_file_path + x_train_val_file_name, train_val_data_type)
    save_model_data(y_train_val, data_file_path + y_train_val_file_name, train_val_data_type)

    from src.base.config import val_data_type
    from src.base.config import x_val_file_name, y_val_file_name
    save_model_data(x_val, data_file_path + x_val_file_name, val_data_type)
    save_model_data(y_val, data_file_path + y_val_file_name, val_data_type)
    return x_train_val, y_train_val


def export_train_test_data(x_data, y_data):
    x_data = x_data[0:config.user_id_max]
    show_title("拆分训练数据集和测试数据集")
    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=config.seed, stratify=y_data)
    from src.base.config import train_data_type
    from src.base.config import x_train_file_name, y_train_file_name
    save_model_data(x_train, data_file_path + x_train_file_name, train_data_type)
    save_model_data(y_train, data_file_path + y_train_file_name, train_data_type)

    from src.base.config import test_data_type
    from src.base.config import x_test_file_name, y_test_file_name
    save_model_data(x_test, data_file_path + x_test_file_name, test_data_type)
    save_model_data(y_test, data_file_path + y_test_file_name, test_data_type)
    return x_train, y_train


def export_generate_data(lst_data):
    # from src.data.generate_data import generate_fix_data
    # x_data = generate_fix_data(lst_data)

    # from generate_data import generate_data_no_interval_with_repeat
    # x_data, y_data, x_w2v = generate_data_no_interval_with_repeat(x_csv, y_csv)
    return ExportDataTypeGenerateFunc[config.export_data_type](lst_data)


def export_base_data():
    lst_data, y_data = export_day_list_data()
    # from src.data.load_data import load_day_list_data
    # lst_data, y_data = load_day_list_data()
    export_w2v_data(lst_data)
    x_data = export_generate_data(lst_data)
    return x_data, y_data


def export_day_list_data():
    from src.data.load_data import load_original_data
    show_title("加载原始数据")
    x_csv, y_csv = load_original_data()

    from src.data.generate_data import generate_day_list_data
    show_title("导出每个用户每天访问数据的不截断列表")
    lst_data, y_data = generate_day_list_data(x_csv, y_csv)
    save_model_data(lst_data, data_file_path + config.lst_data_file_name, config.base_data_type)
    save_model_data(y_data, data_file_path + config.y_data_file_name, config.base_data_type)
    return lst_data, y_data


def export_w2v_data(lst_data):
    show_title("导出用于Word2Vec训练的数据")
    from src.data.generate_data import generate_w2v_data
    x_w2v = generate_w2v_data(lst_data)
    from src.base.config import data_w2v_path, w2v_file_name, w2v_data_type
    save_model_data(x_w2v, data_w2v_path + w2v_file_name, w2v_data_type)


def export_day_fix_sequence(lst_data):
    show_title("加工数据为 91 天的序列数据，每天为定长的数据序列")
    pass


def export_day_statistical_sequence(lst_data):
    from src.data.generate_data import generate_day_statistical_sequence
    show_title("加工数据为 91 天的序列数据，每天为6个特征(最大值、最小值、平均值)96维数据")
    x_data = generate_day_statistical_sequence(lst_data)
    from src.base.config import x_data_file_name, base_data_type
    save_model_data(x_data, data_file_path + x_data_file_name, base_data_type)
    return x_data


def export_week_fix_sequence(lst_data):
    pass


def export_week_statistical_sequence(lst_data):
    pass


def export_user_fix_sequence(lst_data):
    show_title("加工数据为无间隔有重复的数据列表")
    pass


def export_user_statistical_sequence(lst_data):
    pass


ExportDataTypeGenerateFunc = {
        ExportDataType.day_fix_sequence: export_day_fix_sequence,
        ExportDataType.day_statistical_sequence: export_day_statistical_sequence,
        ExportDataType.week_fix_sequence: export_week_fix_sequence,
        ExportDataType.week_statistical_sequence: export_week_statistical_sequence,
        ExportDataType.user_fix_sequence: export_user_fix_sequence,
        ExportDataType.user_statistical_sequence: export_user_statistical_sequence
}

# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   Export_tf_idf_Data.py
@Version    :   v0.1
@Time       :   2020-07-09 18:06
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   输出训练使用的数据集
@理解：
"""

from sklearn.model_selection import train_test_split

# common imports
import config
from config import data_file_path, train_data_type
from generate_data import generate_balance_data
from load_data import load_model_data
from save_data import save_model_data
from tools import beep_end, show_title


# ----------------------------------------------------------------------
def export_balance_set():
    """
    使用已经导出的文件，单独导出平衡数据集
    """
    x_train = load_model_data(data_file_path + 'x_train', train_data_type)
    y_train = load_model_data(data_file_path + 'y_train', train_data_type)

    show_title("平衡训练数据集({})的类别".format(config.label_name))
    x_train_balance, y_train_balance = generate_balance_data(x_train, y_train)
    save_model_data(x_train_balance, config.data_file_path + 'x_train_balance', '平衡的训练数据集')
    save_model_data(y_train_balance, config.data_file_path + 'y_train_balance', '平衡的训练数据集')

    x_train_val = load_model_data(data_file_path + 'x_train_val', train_data_type)
    y_train_val = load_model_data(data_file_path + 'y_train_val', train_data_type)

    show_title("平衡拆分验证的训练数据集({})的类别".format(config.label_name))
    x_train_val_balance, y_train_val_balance = generate_balance_data(x_train_val, y_train_val)
    save_model_data(x_train_val_balance, config.data_file_path + 'x_train_val_balance', '平衡拆分验证的训练数据集')
    save_model_data(y_train_val_balance, config.data_file_path + 'y_train_val_balance', '平衡拆分验证的训练数据集')


# ----------------------------------------------------------------------
def export_all_data_set():
    """
    一次性导出所有的数据
    """
    from load_data import load_original_data
    show_title("加载原始数据")
    x_csv, y_csv = load_original_data()

    from generate_data import generate_day_list_data
    show_title("生成每个用户每天访问数据的不截断列表")
    x_data, y_data = generate_day_list_data(x_csv, y_csv)

    # show_title("加工数据为定长的数据列表")
    # from generate_data import generate_fix_data
    # x_data, y_data, x_w2v = generate_fix_data(x_csv, y_csv)

    # show_title("加工数据为无间隔有重复的数据列表")
    # from generate_data import generate_data_no_interval_with_repeat
    # x_data, y_data, x_w2v = generate_data_no_interval_with_repeat(x_csv, y_csv)

    from generate_data import generate_w2v_data
    show_title("保存用于Word2Vec训练的数据")
    x_w2v = generate_w2v_data(x_data)
    save_model_data(x_w2v, config.data_w2v_path + 'x_w2v', 'w2v数据集')

    show_title("拆分训练数据集和测试数据集")
    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=config.seed, stratify=y_data)
    save_model_data(x_train, config.data_file_path + 'x_train', '训练数据集')
    save_model_data(y_train, config.data_file_path + 'y_train', '训练数据集')
    save_model_data(x_test, config.data_file_path + 'x_test', '测试数据集')
    save_model_data(y_test, config.data_file_path + 'y_test', '测试数据集')

    show_title("拆分训练数据集和验证数据集")
    x_train_val, x_val, y_train_val, y_val = train_test_split(
            x_train, y_train, random_state=config.seed, stratify=y_train)
    save_model_data(x_train_val, config.data_file_path + 'x_train_val', '无验证训练数据集')
    save_model_data(y_train_val, config.data_file_path + 'y_train_val', '无验证训练数据集')
    save_model_data(x_val, config.data_file_path + 'x_val', '验证数据集')
    save_model_data(y_val, config.data_file_path + 'y_val', '验证数据集')

    show_title("平衡训练数据集({})的类别".format(config.label_name))
    x_train_balance, y_train_balance = generate_balance_data(x_train, y_train)
    save_model_data(x_train_balance, config.data_file_path + 'x_train_balance', '平衡的训练数据集')
    save_model_data(y_train_balance, config.data_file_path + 'y_train_balance', '平衡的训练数据集')

    show_title("平衡拆分验证的训练数据集({})的类别".format(config.label_name))
    x_train_val_balance, y_train_val_balance = generate_balance_data(x_train_val, y_train_val)
    save_model_data(x_train_val_balance, config.data_file_path + 'x_train_val_balance', '平衡拆分验证的训练数据集')
    save_model_data(y_train_val_balance, config.data_file_path + 'y_train_val_balance', '平衡拆分验证的训练数据集')

    show_title("数据清洗完成！")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    export_all_data_set()
    # export_balance_set()
    beep_end()  # 运行结束的提醒

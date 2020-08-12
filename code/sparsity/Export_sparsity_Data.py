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
# common imports
import config

from config import creative_id_step_size
from generate_data import generate_fix_data
from load_data import load_original_data
from tools import beep_end


def export_word2vec_data():
    from load_data import load_word2vec_file
    from generate_data import generate_word2vec_data_with_interval
    from generate_data import generate_word2vec_data_no_interval
    from show_data import show_word2vec_data
    from save_data import save_word2vec_data

    x_csv, _ = load_original_data()
    x_creative_id = generate_word2vec_data_no_interval(x_csv)

    field_list = [  # 输入数据处理：选择需要的列
        "user_id",  # 0
        "creative_id_inc_sparsity",  # 1
        "time_id",  # 2
    ]
    x_csv = load_word2vec_file('../../save_data/sparsity/train_data_all_sparsity_v.csv', field_list)

    creative_id_window = creative_id_step_size * 1
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    save_data_path = '../../save_data/sparsity/no_interval/word2vec/'
    x_creative_id = generate_word2vec_data_no_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    save_data_path = '../../save_data/sparsity/with_interval/word2vec/'
    x_creative_id = generate_word2vec_data_with_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    creative_id_window = creative_id_step_size * 3
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    save_data_path = '../../save_data/sparsity/no_interval/word2vec/'
    x_creative_id = generate_word2vec_data_no_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    save_data_path = '../../save_data/sparsity/with_interval/word2vec/'
    x_creative_id = generate_word2vec_data_with_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    print("\n数据清洗完成！")


# ----------------------------------------------------------------------
def export_data_set():
    from generate_data import generate_data_no_interval_with_repeat
    from load_data import load_original_data
    from sklearn.model_selection import train_test_split
    from generate_data import generate_balance_data
    from save_data import save_data

    print('-' * 5 + '>' * 3 + "加载原始数据" + '<' * 3 + '-' * 5)
    x_csv, y_csv = load_original_data()

    print('-' * 5 + '>' * 3 + "加工数据为定长的数据列表" + '<' * 3 + '-' * 5)
    x_data, y_data, x_w2v = generate_fix_data(x_csv, y_csv)
    save_data(x_w2v, config.data_w2v_path + 'x_w2v', 'w2v数据集')

    # print('-' * 5 + "   加工数据为无间隔有重复的数据列表   " + '-' * 5)
    # x_data, y_data, x_w2v = generate_data_no_interval_with_repeat(x_csv, y_csv)
    # save_data(x_w2v, config.data_w2v_path + 'x_w2v', 'w2v数据集')
    #
    print('-' * 5 + "   拆分训练数据集和测试数据集   " + '-' * 5)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=config.seed, stratify=y_data)
    save_data(x_train, config.data_file_path + 'x_train', '训练数据集')
    save_data(y_train, config.data_file_path + 'y_train', '训练数据集')
    save_data(x_test, config.data_file_path + 'x_test', '测试数据集')
    save_data(y_test, config.data_file_path + 'y_test', '测试数据集')

    print('-' * 5 + "   平衡训练数据集({})的类别   ".format(config.label_name) + '-' * 5)
    x_train_balance, y_train_balance = generate_balance_data(x_train, y_train)
    save_data(x_train_balance, config.data_file_path + 'x_train_balance', '平衡的训练数据集')
    save_data(y_train_balance, config.data_file_path + 'y_train_balance', '平衡的训练数据集')

    print('-' * 5 + "   拆分训练数据集和验证数据集   " + '-' * 5)
    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, random_state=config.seed, stratify=y_train)
    save_data(x_train_val, config.data_file_path + 'x_train_val', '无验证训练数据集')
    save_data(y_train_val, config.data_file_path + 'y_train_val', '无验证训练数据集')
    save_data(x_val, config.data_file_path + 'x_val', '验证数据集')
    save_data(y_val, config.data_file_path + 'y_val', '验证数据集')

    print('-' * 5 + "   平衡拆分验证的训练数据集({})的类别   ".format(config.label_name) + '-' * 5)
    x_train_val_balance, y_train_val_balance = generate_balance_data(x_train_val, y_train_val)
    save_data(x_train_val_balance, config.data_file_path + 'x_train_val_balance', '平衡拆分验证的训练数据集')
    save_data(y_train_val_balance, config.data_file_path + 'y_train_val_balance', '平衡拆分验证的训练数据集')

    print("\n数据清洗完成！")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    # export_word2vec_data()
    export_data_set()  # 运行结束的提醒

    beep_end()

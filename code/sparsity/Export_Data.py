# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   Export_Data.py
@Version    :   v0.1
@Time       :   2020-07-09 18:06
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   输出训练使用的数据集
@理解：
"""
# common imports
import os

import winsound
from config import creative_id_step_size


# ----------------------------------------------------------------------
def export_word2vec_data():
    from load_data import load_word2vec_file
    from generate_data import generate_word2vec_data_with_interval
    from generate_data import generate_word2vec_data_no_interval
    from show_data import show_word2vec_data
    from save_data import save_word2vec_data

    field_list = [  # 输入数据处理：选择需要的列
        "user_id",  # 0
        "creative_id_inc_sparsity",  # 1
        "time_id",  # 2
    ]
    x_csv = load_word2vec_file('../../save_data/sparsity/train_data_all_sparsity_v.csv', field_list)

    creative_id_window = creative_id_step_size * 5
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    save_data_path = '../../save_data/sparsity/no_interval/with_repeat/word2vec/'
    x_creative_id = generate_word2vec_data_with_interval(x_csv, creative_id_begin, creative_id_end)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    save_data_path = '../../save_data/sparsity/no_interval/no_repeat/word2vec/'
    x_creative_id = generate_word2vec_data_no_interval(x_csv, creative_id_begin, creative_id_end)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    creative_id_window = creative_id_step_size * 8
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    save_data_path = '../../save_data/sparsity/no_interval/with_repeat/word2vec/'
    x_creative_id = generate_word2vec_data_with_interval(x_csv, creative_id_begin, creative_id_end)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    save_data_path = '../../save_data/sparsity/no_interval/no_repeat/word2vec/'
    x_creative_id = generate_word2vec_data_no_interval(x_csv, creative_id_begin, creative_id_end)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    print("\n数据清洗完成！")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == '__main__':
    export_word2vec_data()
    # 运行结束的提醒
    winsound.Beep(600, 500)

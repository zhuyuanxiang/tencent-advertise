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
import winsound
from config import creative_id_step_size, seed


# ----------------------------------------------------------------------
def export_word2vec_data():
    from src.无用代码.load_data import load_word2vec_file
    from src.无用代码.generate_data import generate_word2vec_data_with_interval
    from src.无用代码.generate_data import generate_word2vec_data_no_interval
    from src.data.show_data import show_word2vec_data
    from src.无用代码.save_data import save_word2vec_data

    field_list = [  # 输入数据处理：选择需要的列
        "user_id",  # 0
        "creative_id_inc_tf_idf",  # 1
        "time_id",  # 2
    ]
    x_csv = load_word2vec_file('../../save_data/tf_idf/train_data_all_tf_idf_v.csv', field_list)

    creative_id_window = creative_id_step_size * 5
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    save_data_path = '../../save_data/tf_idf/no_interval/word2vec/'
    x_creative_id = generate_word2vec_data_no_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    save_data_path = '../../save_data/tf_idf/with_interval/word2vec/'
    x_creative_id = generate_word2vec_data_with_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    creative_id_window = creative_id_step_size * 8
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window

    save_data_path = '../../save_data/tf_idf/no_interval/word2vec/'
    x_creative_id = generate_word2vec_data_no_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    save_data_path = '../../save_data/tf_idf/with_interval/word2vec/'
    x_creative_id = generate_word2vec_data_with_interval(x_csv)
    show_word2vec_data(x_creative_id)
    save_word2vec_data(x_creative_id, creative_id_window, save_data_path)
    del x_creative_id

    print("\n数据清洗完成！")


# ----------------------------------------------------------------------
def export_data_set():
    from src.无用代码.generate_data import generate_data_no_interval_with_repeat
    from src.data.load_data import load_original_data
    from src.无用代码.save_data import save_data_set
    from sklearn.model_selection import train_test_split

    field_list = [  # 输入数据处理：选择需要的列
        "user_id",
        "creative_id_inc_tf_idf",
        "time_id",
        "click_times",
    ]
    label_name = 'gender'
    label_list = ['age', 'gender']
    load_file_path = '../../save_data/tf_idf/'
    save_file_path = '../../save_data/tf_idf/no_interval/with_repeat/'

    creative_id_window = creative_id_step_size * 1
    creative_id_begin = 0
    creative_id_end = creative_id_begin + creative_id_window

    x_csv, y_csv = load_original_data()
    x_data, y_data = generate_data_no_interval_with_repeat(x_csv, y_csv)

    label_data = y_data[:, label_list.index(label_name)]
    x_train, x_test, y_train, y_test = train_test_split(x_data, label_data, random_state=seed, stratify=label_data)
    del x_data, y_data

    save_data_set(x_train, y_train, x_test, y_test)
    del x_train, y_train, x_test, y_test

    print("\n数据清洗完成！")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    # export_word2vec_data()
    export_data_set()
    # 运行结束的提醒
    winsound.Beep(600, 500)

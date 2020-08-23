# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   data.py
@Version    :   v0.1
@Time       :   2020-08-15 11:48
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import pickle

import numpy as np

from src.base import tools

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
from src.base.config import data_file_path
from src.data.show_data import show_example_data
from src.base.tools import show_title

if __name__ == '__main__':
    # 运行结束的提醒
    tools.beep_end()


def save_data_set(x_train, y_train, x_test, y_test):
    '''
    保存用于训练和测试模型时使用的数据集，缩短模型调优时载入数据的时间，固定模型训练的数据，方便模型对比
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    '''
    data_type = "训练数据集"
    print('-' * 5 + "   保存{0}：{1}   ".format(data_type, len(y_train)) + '-' * 5)
    show_example_data(x_train, y_train, data_type)
    np.save(data_file_path + 'x_train', x_train)
    np.save(data_file_path + 'y_train', y_train)
    print(data_type + "保存成功。")

    data_type = "测试数据集"
    print('-' * 5 + "   保存{0}：{1}   ".format(data_type, len(y_train)) + '-' * 5)
    show_example_data(x_test, y_test, data_type)
    np.save(data_file_path + 'x_test', x_test)
    np.save(data_file_path + 'y_test', y_test)
    print(data_type + "保存成功。")


def save_word2vec_data(x_creative_id, creative_id_window, file_path):
    """
    保存训练 word2vec 用的数据
    :param x_creative_id:
    :param creative_id_window:
    :param file_path:
    :return:
    """
    file_name = file_path + 'creative_id_{0}'.format(creative_id_window)
    show_title("保存数据集:{0}".format(file_name))
    with open(file_name, 'wb') as fname:
        pickle.dump(x_creative_id, fname, -1)
    print("Word2Vec 数据保存成功")

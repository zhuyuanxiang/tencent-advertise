# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
--------------------------------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   save_data.py
@Version    :   v0.1
@Time       :   2020-06-07 15:41
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :   数据保存模块
@理解：
"""

import pickle


# --------------------------------------------------
def save_word2vec_data_with_interval(x_creative_id, creative_id_window):
    file_name = 'save_data/word2vec/with_interval/creative_id_{0}'.format(creative_id_window)
    print('-' * 5 + "   保存数据集:{0}   ".format(file_name) + '-' * 5)
    fname = open(file_name, 'wb')
    pickle.dump(x_creative_id, fname, -1)
    fname.close()
    print("数据保存成功")


# --------------------------------------------------
def save_word2vec_data_no_interval(x_creative_id, creative_id_window):
    file_name = 'save_data/word2vec/no_interval/creative_id_{0}'.format(creative_id_window)
    print('-' * 5 + "   保存数据集:{0}   ".format(file_name) + '-' * 5)
    fname = open(file_name, 'wb')
    pickle.dump(x_creative_id, fname, -1)
    fname.close()
    print("数据保存成功")

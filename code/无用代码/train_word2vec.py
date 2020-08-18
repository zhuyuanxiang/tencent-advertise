# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   Tencent-Advertise
@File       :   Word2Vec.py
@Version    :   v0.1
@Time       :   2020-06-26 18:54
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   训练 Word2Vec 模块
@理解：
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import winsound

from gensim.models import Word2Vec

seed = 42


# ----------------------------------------------------------------------
def train_word2vec_model_with_gensim(words_lists):
    print('-' * 5 + "   训练 word2vec({0}_{1}) 模型   ".format(embedding_size, embedding_window) + '-' * 5)
    model = Word2Vec(
        words_lists,
        size=embedding_size,
        window=embedding_window,
        min_count=1,
        seed=seed,
        workers=8,
        sg=0,  # 0:CBOW;1:Skip-Gram
        iter=10,
        sorted_vocab=False,
        batch_words=4096
    )
    return model


def load_words_list(words_path):
    file_name = words_path + file_prefix + '_{0}'.format(creative_id_window)
    print('-' * 5 + "   载入数据集: {0}   ".format(file_name) + '-' * 5)
    fname = open(file_name, 'rb')
    words_lists: list = pickle.load(fname)
    for one_list in words_lists:
        for j, number in enumerate(one_list):
            one_list[j] = number
    return words_lists


def save_word2vec_model(model, file_name):
    print('-' * 5 + "   保存 word2vec 模型   " + '-' * 5)
    model.save(file_name + '.model')
    model.wv.save(file_name + '.kv')


def main(path, window):
    word2vec_words_lists = load_words_list('save_data/' + path)
    word2vec_model = train_word2vec_model_with_gensim(word2vec_words_lists)
    save_word2vec_model(word2vec_model,
                        'save_model/' + path + file_prefix + "_{0}_{1}_{2}".format(embedding_size, embedding_window, creative_id_window))


# =====================================================
if __name__ == '__main__':
    # 清洗数据需要的变量
    user_id_max = 900000  # 用户数
    creative_id_max = 2481135 - 1  # 最大的素材编号 = 素材的总数量 - 1，这个编号已经修正了数据库与Python索引的区别
    click_times_max = 152  # 所有素材中最大的点击次数
    time_id_max = 91
    creative_id_step_size = 128000
    creative_id_window = creative_id_step_size * 5
    embedding_size = 32
    embedding_window = 5
    file_prefix = 'creative_id'
    no_interval_path = 'word2vec/no_interval/'
    main(no_interval_path, creative_id_step_size * 5)
    main(no_interval_path, creative_id_step_size * 8)

    with_interval_path = 'word2vec/with_interval/'
    main(with_interval_path, creative_id_step_size * 5)
    main(with_interval_path, creative_id_step_size * 8)

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

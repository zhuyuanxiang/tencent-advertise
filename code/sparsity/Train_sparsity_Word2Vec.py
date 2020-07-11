# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   Tencent-Advertise
@File       :   Train_sparsity_Word2Vec.py
@Version    :   v0.1
@Time       :   2020-06-26 18:54
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   训练 Word2Vec 模块
@理解：
"""
import matplotlib.pyplot as plt
import pickle
import winsound

from gensim.models import Word2Vec

# ----------------------------------------------------------------------
from config import seed, creative_id_step_size


def train_word2vec_model_with_gensim(words_lists):
    print('-' * 5 + "   训练 word2vec({0}_{1}) 模型   ".format(embedding_size, embedding_window) + '-' * 5)
    model = Word2Vec(
        words_lists,
        size=embedding_size,
        window=embedding_window,
        min_count=1,
        seed=seed,
        workers=8,
        sg=0,  # 0:CBOW; 1:Skip-Gram
        iter=20,
        sorted_vocab=False,
        batch_words=4096
    )
    return model


def train_word2vec_model_with_tensorflow(words_lists, size, window, seed=seed, sg=0, iter=5, batch_words=4096):
    pass


def load_words_list(words_path, creative_id_window):
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


def main(path, creative_id_window):
    data_path = '../../save_data' + path
    word2vec_words_lists = load_words_list(data_path, creative_id_window)
    word2vec_model = train_word2vec_model_with_gensim(word2vec_words_lists)
    file_name = '../../save_model' + path + file_prefix + "_{0}_{1}_{2}".format(embedding_size, embedding_window, creative_id_window)
    save_word2vec_model(word2vec_model, file_name)


# =====================================================
if __name__ == '__main__':
    # 清洗数据需要的变量
    embedding_size = 32
    embedding_window = 5
    file_prefix = 'creative_id'
    no_interval_path = '/sparsity/no_interval/word2vec/'
    main(no_interval_path, creative_id_step_size * 1)
    main(no_interval_path, creative_id_step_size * 3)

    with_interval_path = '/sparsity/with_interval/word2vec/'
    main(with_interval_path, creative_id_step_size * 1)
    main(with_interval_path, creative_id_step_size * 3)

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

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
import winsound
# ----------------------------------------------------------------------
from gensim.models import Word2Vec
from load_data import load_data
from config import embedding_size, embedding_window, seed, data_w2v_path, model_w2v_path
from save_data import save_word2vec_weights


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


# =====================================================
if __name__ == '__main__':
    x_w2v = load_data(data_w2v_path + 'x_w2v')
    model_word2vec = train_word2vec_model_with_gensim(x_w2v)
    save_word2vec_weights(model_word2vec)
    # 运行结束的提醒
    winsound.Beep(600, 500)

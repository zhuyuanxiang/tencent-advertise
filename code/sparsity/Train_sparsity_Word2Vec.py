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
# ----------------------------------------------------------------------

from tools import show_title, beep_end


def train_word2vec_model_with_gensim(words_lists):
    from gensim.models import Word2Vec
    from config import embedding_size, embedding_window, seed
    show_title(f"训练 word2vec({embedding_size}_{embedding_window}) 模型")
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
    # TODO: 使用GPU训练
    pass


# =====================================================
if __name__ == '__main__':
    from config import data_w2v_path
    from code.data.load_data import load_model_data
    from code.data.save_data import save_word2vec_weights

    x_w2v = load_model_data(data_w2v_path + 'x_w2v')
    model_word2vec = train_word2vec_model_with_gensim(x_w2v)
    save_word2vec_weights(model_word2vec.wv)
    beep_end()

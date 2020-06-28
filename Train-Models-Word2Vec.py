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
@Desc       :   
@理解：
"""

import math
import os
import pickle
import random
import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import winsound

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tensorflow import keras
from tensorflow_core.python.keras import optimizers, losses, metrics
from tensorflow_core.python.keras.layers import Embedding, GlobalMaxPooling1D, Dense
from keras_preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.regularizers import l2


# ----------------------------------------------------------------------
# 输出训练的结果
def output_result(results, predictions):
    print("模型预测-->", end='')
    print("损失值 = {}，精确度 = {}".format(results[0], results[1]))
    if label_name == 'age':
        np_argmax = np.argmax(predictions, 1)
        # print("前 30 个真实的预测数据 =", np.array(X_test[:30], dtype = int))
        print("前 30 个真实的目标数据 =", np.array(y_test[:30], dtype=int))
        print("前 30 个预测的目标数据 =", np.array(np.argmax(predictions[:30], 1), dtype=int))
        print("前 30 个预测的结果数据 =", )
        print(predictions[:30])
        for i in range(10):
            print("类别 {0} 的真实数目：{1}，预测数目：{2}".format(i, sum(y_test == i), sum(np_argmax == i)))
    elif label_name == 'gender':
        predict_gender = np.array(predictions > 0.5, dtype=int)
        print("sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) = error% =",
              sum(abs(predict_gender - y_test)) / sum(y_test) * 100, '%')
        print("前100个真实的目标数据 =", np.array(y_test[:100], dtype=int))
        print("前100个预测的目标数据 =", np.array(predict_gender[:100], dtype=int))
        print("sum(predictions>0.5) =", sum(predict_gender))
        print("sum(y_test) =", sum(y_test))
        print("sum(abs(predictions-y_test))=error_number=", sum(abs(predict_gender - y_test)))
    else:
        print("错误的标签名称：", label_name)
        pass
    pass


print('>' * 15 + ' ' * 3 + "train_single_gender" + ' ' * 3 + '<' * 15)
# 定义全局定制变量
RMSProp_lr = 5e-04
batch_size = 256
embedding_size = 32
epochs = 40
label_name = 'gender'
max_len = 128  # 64:803109，128:882952 个用户；64：1983350，128：2329077 个素材
# 定制 素材库大小 = creative_id_end - creative_id_start = creative_id_num = creative_id_step_size * (1 + 3 + 1)
creative_id_step_size = 128000
creative_id_window = creative_id_step_size * 5
# ----------------------------------------------------------------------
print('-' * 5 + ' ' * 3 + "加载 word2vec 模型" + ' ' * 3 + '-' * 5)
word2vec = KeyedVectors.load('save_model/word2vec/vectors.kv')
embedding_weights = np.zeros((creative_id_window, embedding_size))
for word, index in word2vec.vocab.items():
    try:
        embedding_weights[int(word), :] = word2vec[word]
    except KeyError:
        pass
pass

model = keras.Sequential()
model.add(Embedding(creative_id_window, embedding_size, input_length=max_len, weights=[embedding_weights]))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
model.summary()
print("保存模型的原始结构：", model.save('save_model/word2vec/word2vec_m0_'+label_name+'.h5'))
model.compile(optimizer=optimizers.RMSprop(lr=RMSProp_lr),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# ----------------------------------------------------------------------
# 加载数据
print('-' * 5 + ' ' * 3 + "加载数据集" + ' ' * 3 + '-' * 5)
X_train = pad_sequences(np.load('save_data/no_time_no_repeat/x_train_no_time_no_repeat_' + label_name + '.npy',
                                allow_pickle=True)[:, 0],
                        maxlen=max_len,
                        padding='post')
y_train = np.load('save_data/no_time_no_repeat/y_train_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)
X_test = pad_sequences(np.load('save_data/no_time_no_repeat/x_test_no_time_no_repeat_' + label_name + '.npy',
                               allow_pickle=True)[:, 0],
                       maxlen=max_len,
                       padding='post')
y_test = np.load('save_data/no_time_no_repeat/y_test_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)
# ----------------------------------------------------------------------
# 训练网络模型
# 使用验证集
print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, use_multiprocessing=True, verbose=2)
print("保存第一次模型训练的权重", model.save_weights('save_model/word2vec/word2vec_m1_'+label_name+'.bin'))
f = open('save_model/word2vec/word2vec_m1_'+label_name+'.pkl', 'wb')
pickle.dump(history, f)
f.close()
results = model.evaluate(X_test, y_test, verbose=0)
predictions = model.predict(X_test).squeeze()
output_result(results, predictions)

# ----------------------------------------------------------------------
# 不使用验证集，训练次数减半
print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
history = model.fit(X_train, y_train, epochs=epochs // 2, batch_size=batch_size,
                    use_multiprocessing=True, verbose=2)
print("保存第二次模型训练的权重", model.save_weights('save_model/word2vec/word2vec_m2'+label_name+'.bin'))
f = open('save_model/word2vec/word2vec_m2'+label_name+'.pkl', 'wb')
pickle.dump(history.history, f)
f.close()
results = model.evaluate(X_test, y_test, verbose=0)
predictions = model.predict(X_test).squeeze()
output_result(results, predictions)
# =====================================================
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

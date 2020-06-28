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

from gensim.models import KeyedVectors
from tensorflow import keras
from tensorflow_core.python.keras import optimizers, losses, metrics, Input, Model
from tensorflow_core.python.keras.layers import Embedding, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling1D, concatenate, Activation, Conv1D, MaxPooling1D, AveragePooling1D, RNN, LSTM
from keras_preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.regularizers import l2
from tensorflow_core.python.keras.utils import plot_model


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
        print(
            "sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) = error% =",
            sum(abs(predict_gender - y_test)) / sum(y_test) * 100, '%'
        )
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
batch_size = 1024
embedding_size = 32
epochs = 30
label_name = 'gender'
max_len = 128  # 64:803109，128:882952 个用户；64：1983350，128：2329077 个素材
period_days = 7
period_length = 21  # 每个周期的素材数目
time_id_max = 91
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
embedding_weights[0, :] = np.zeros(embedding_size)


def construct_keras_model():
    keras_model = keras.Sequential()
    keras_model.add(
        Embedding(
            creative_id_window, embedding_size, input_length=max_len, weights=[embedding_weights], trainable=False
        )
    )
    # keras_model.add(GlobalMaxPooling1D())
    keras_model.add(GlobalAveragePooling1D())
    # keras_model.add(Dropout(0.5))
    # keras_model.add(BatchNormalization())
    keras_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    # keras_model.add(Dropout(0.5))
    # keras_model.add(BatchNormalization())
    keras_model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
    keras_model.summary()
    # print("保存模型的原始结构：", keras_model.save('save_model/word2vec/word2vec_m0_' + label_name + '.h5'))
    keras_model.compile(
        optimizer=optimizers.RMSprop(lr=RMSProp_lr), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy]
    )
    return keras_model


def construct_keras_api_model():
    input_no_time_no_repeat = Input(shape=max_len, dtype='int32', name='no_time_no_repeat')
    embedded_no_time_no_repeat = Embedding(
        creative_id_window,
        embedding_size,
        weights=[embedding_weights],
        name='Embedded_no_time_no_repeat',
        trainable=False
    )(input_no_time_no_repeat)

    # ----------------------------------------------------------------------
    GM_creative_id = GlobalMaxPooling1D(name='GM_creative_id')(embedded_no_time_no_repeat)

    GM_x = Dense(embedding_size, kernel_regularizer=l2(0.001), name='GM_Dense_0101')(GM_creative_id)
    GM_x = BatchNormalization(name='GM_Dense_BN_0101')(GM_x)
    GM_x = Activation('relu', name='GM_Dense_Activation_0101')(GM_x)

    GM_x = Dropout(0.5, name='GM_Dense_Dropout_0201')(GM_x)
    GM_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001), name='GM_Dense_0201')(GM_x)
    GM_x = BatchNormalization(name='GM_Dense_BN_0201')(GM_x)
    GM_x = Activation('relu', name='GM_Dense_Activation_0201')(GM_x)

    GM_x = Dense(1, 'sigmoid', name='GM_Output')(GM_x)

    # ----------------------------------------------------------------------
    GA_creative_id = GlobalAveragePooling1D(name='GA_creative_id')(embedded_no_time_no_repeat)

    GA_x = Dense(embedding_size, kernel_regularizer=l2(0.001), name='GA_Dense_0101')(GA_creative_id)
    GA_x = BatchNormalization(name='GA_Dense_BN_0101')(GA_x)
    GA_x = Activation('relu', name='GA_Dense_Activation_0101')(GA_x)

    GA_x = Dropout(0.5, name='GA_Dense_Dropout_0201')(GA_x)
    GA_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001), name='GA_Dense_0201')(GA_x)
    GA_x = BatchNormalization(name='GA_Dense_BN_0201')(GA_x)
    GA_x = Activation('relu', name='GA_Dense_Activation_0201')(GA_x)

    GA_x = Dense(1, 'sigmoid', name='GA_Output')(GA_x)

    # ==================================================================================
    input_fix_creative_id = Input(
        shape=(time_id_max * period_length // period_days), dtype='int32', name='fix_creative_id'
    )
    embedded_fix_creative_id = Embedding(
        creative_id_window,
        embedding_size,
        weights=[embedding_weights],
        name='Embedded_fix_creative_id',
        trainable=False
    )(input_fix_creative_id)

    # ----------------------------------------------------------------------
    Conv_GM_creative_id = Conv1D(
        embedding_size, 14, 7, activation='relu', name='Conv_GM_Conv_0101'
    )(embedded_fix_creative_id)
    Conv_GM_x = MaxPooling1D(2, name='Conv_GM_Conv_MaxPool_0101')(Conv_GM_creative_id)
    Conv_GM_x = Conv1D(embedding_size, 3, 2, activation='relu', name='Conv_GM_Conv_0102')(Conv_GM_x)
    Conv_GM_x = GlobalMaxPooling1D(name='Conv_GM_GM')(Conv_GM_x)

    Conv_GM_x = Dense(embedding_size, kernel_regularizer=l2(0.001), name='Conv_GM_Dense_0101')(Conv_GM_x)
    Conv_GM_x = BatchNormalization(name='Conv_GM_Dense_BN_0101')(Conv_GM_x)
    Conv_GM_x = Activation('relu', name='Conv_GM_Dense_Activation_0101')(Conv_GM_x)

    Conv_GM_x = Dropout(0.5, name='Conv_GM_Dense_Dropout_0201')(Conv_GM_x)
    Conv_GM_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001), name='Conv_GM_Dense_02Conv')(Conv_GM_x)
    Conv_GM_x = BatchNormalization(name='Conv_GM_Dense_BN_0201')(Conv_GM_x)
    Conv_GM_x = Activation('relu', name='Conv_GM_Dense_Activation_0201')(Conv_GM_x)

    Conv_GM_x = Dense(1, 'sigmoid', name='Conv_GM_Output')(Conv_GM_x)

    # ----------------------------------------------------------------------
    Conv_GA_creative_id = Conv1D(
        embedding_size, 14, 7, activation='relu', name='Conv_GA_Conv_0101'
    )(embedded_fix_creative_id)
    Conv_GA_x = AveragePooling1D(2, name='Conv_GA_Conv_AveragePool_0101')(Conv_GA_creative_id)
    Conv_GA_x = Conv1D(embedding_size, 3, 2, activation='relu', name='Conv_GA_Conv_0102')(Conv_GA_x)
    Conv_GA_x = GlobalAveragePooling1D(name='Conv_GA_GA')(Conv_GA_x)

    Conv_GA_x = Dense(embedding_size, kernel_regularizer=l2(0.001), name='Conv_GA_Dense_0101')(Conv_GA_x)
    Conv_GA_x = BatchNormalization(name='Conv_GA_Dense_BN_0101')(Conv_GA_x)
    Conv_GA_x = Activation('relu', name='Conv_GA_Dense_Activation_0101')(Conv_GA_x)

    Conv_GA_x = Dropout(0.5, name='Conv_GA_Dense_Dropout_0201')(Conv_GA_x)
    Conv_GA_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001), name='Conv_GA_Dense_02Conv')(Conv_GA_x)
    Conv_GA_x = BatchNormalization(name='Conv_GA_Dense_BN_0201')(Conv_GA_x)
    Conv_GA_x = Activation('relu', name='Conv_GA_Dense_Activation_0201')(Conv_GA_x)

    Conv_GA_x = Dense(1, 'sigmoid', name='Conv_GA_Output')(Conv_GA_x)

    # ==================================================================================
    input_no_time_with_repeat = Input(shape=max_len, dtype='int32', name='no_time_with_repeat')
    embedded_no_time_with_repeat = Embedding(
        creative_id_window,
        embedding_size,
        weights=[embedding_weights],
        name='Embedded_no_time_with_repeat',
        trainable=False
    )(input_no_time_with_repeat)

    # ----------------------------------------------------------------------
    LSTM_creative_id = LSTM(
        embedding_size, return_sequences=True, name='LSTM_creative_id_0101'
    )(embedded_no_time_with_repeat)
    LSTM_x = LSTM(embedding_size, name='LSTM_creative_id_0102')(LSTM_creative_id)
    LSTM_x = Dense(embedding_size, kernel_regularizer=l2(0.001), name='LSTM_Dense_0101')(LSTM_x)
    LSTM_x = BatchNormalization(name='LSTM_Dense_BN_0101')(LSTM_x)
    LSTM_x = Activation('relu', name='LSTM_Dense_Activation_0101')(LSTM_x)

    LSTM_x = Dropout(0.5, name='LSTM_Dense_Dropout_0201')(LSTM_x)
    LSTM_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001), name='LSTM_Dense_0201')(LSTM_x)
    LSTM_x = BatchNormalization(name='LSTM_Dense_BN_0201')(LSTM_x)
    LSTM_x = Activation('relu', name='LSTM_Dense_Activation_0201')(LSTM_x)

    LSTM_x = Dense(1, 'sigmoid', name='LSTM_Output')(LSTM_x)

    # ----------------------------------------------------------------------
    concatenated = concatenate([
        GM_x,
        GA_x,
        Conv_GM_x,
        Conv_GA_x,
        LSTM_x,
    ], axis=-1)
    output_tensor = Dense(1, 'sigmoid', name='Output_Dense')(concatenated)

    keras_api_model = Model([
        input_no_time_no_repeat,
        input_fix_creative_id,
        input_no_time_with_repeat,
    ], output_tensor)
    keras_api_model.summary()
    plot_model(keras_api_model, to_file='my_log_dir/keras_api_model_5.png')
    print('-' * 5 + ' ' * 3 + "编译模型" + ' ' * 3 + '-' * 5)
    keras_api_model.compile(
        optimizer=optimizers.RMSprop(lr=RMSProp_lr), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy]
    )
    return keras_api_model


model = construct_keras_api_model()

# ----------------------------------------------------------------------
# 加载数据
print('-' * 5 + ' ' * 3 + "加载数据集" + ' ' * 3 + '-' * 5)
X_train_no_time_no_repeat = pad_sequences(
    np.load('save_data/no_time_no_repeat/x_train_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)[:, 0],
    maxlen=max_len,
    padding='pre'
)
X_train_no_time_with_repeat = pad_sequences(
    np.load('save_data/no_time_with_repeat/x_train_no_time_with_repeat_' + label_name + '.npy', allow_pickle=True)[:,
    0],
    maxlen=max_len,
    padding='pre'
)
X_train_fix_creative_id = np.load(
    'save_data/fix_7_21_640k/x_train_fix_week_' + label_name + '.npy', allow_pickle=True
)[:, 0]
y_train = np.load('save_data/no_time_no_repeat/y_train_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)
X_test_no_time_no_repeat = pad_sequences(
    np.load('save_data/no_time_no_repeat/x_test_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)[:, 0],
    maxlen=max_len,
    padding='pre'
)
X_test_no_time_with_repeat = pad_sequences(
    np.load('save_data/no_time_with_repeat/x_test_no_time_with_repeat_' + label_name + '.npy', allow_pickle=True)[:, 0],
    maxlen=max_len,
    padding='pre'
)
X_test_fix_creative_id = np.load(
    'save_data/fix_7_21_640k/x_test_fix_week_' + label_name + '.npy', allow_pickle=True
)[:, 0]
y_test = np.load('save_data/no_time_no_repeat/y_test_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)
# ----------------------------------------------------------------------
# 训练网络模型
# 使用验证集
print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
history = model.fit({
    'no_time_no_repeat': X_train_no_time_no_repeat,
    'fix_creative_id': X_train_fix_creative_id,
    'no_time_with_repeat': X_train_no_time_with_repeat,
},
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    use_multiprocessing=True,
    verbose=2)
# print("保存第一次模型训练的权重", model.save_weights('save_model/word2vec/word2vec_m1_' + label_name + '.bin'))
# f = open('save_model/word2vec/word2vec_m1_' + label_name + '.pkl', 'wb')
# pickle.dump(history.history, f)
# f.close()
results = model.evaluate({
    'no_time_no_repeat': X_test_no_time_no_repeat,
    'fix_creative_id': X_test_fix_creative_id,
    'no_time_with_repeat': X_train_no_time_with_repeat,
},
    y_test,
    verbose=0)
predictions = model.predict({
    'no_time_no_repeat': X_test_no_time_no_repeat,
    'fix_creative_id': X_test_fix_creative_id,
    'no_time_with_repeat': X_test_no_time_with_repeat,
}).squeeze()
output_result(results, predictions)

# ----------------------------------------------------------------------
# 不使用验证集，训练次数减半
print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
history = model.fit({
    'no_time_no_repeat': X_train_no_time_no_repeat,
    'fix_creative_id': X_train_fix_creative_id,
    'no_time_with_repeat': X_train_no_time_with_repeat,
},
    y_train,
    epochs=epochs // 2,
    batch_size=batch_size,
    use_multiprocessing=True,
    verbose=2)
# print("保存第二次模型训练的权重", model.save_weights('save_model/word2vec/word2vec_m2' + label_name + '.bin'))
# f = open('save_model/word2vec/word2vec_m2' + label_name + '.pkl', 'wb')
# pickle.dump(history.history, f)
# f.close()
results = model.evaluate({
    'no_time_no_repeat': X_test_no_time_no_repeat,
    'fix_creative_id': X_test_fix_creative_id,
    'no_time_with_repeat': X_test_no_time_with_repeat,
},
    y_test,
    verbose=0)
predictions = model.predict({
    'no_time_no_repeat': X_test_no_time_no_repeat,
    'fix_creative_id': X_test_fix_creative_id,
    'no_time_with_repeat': X_test_no_time_with_repeat,
}).squeeze()
output_result(results, predictions)
# =====================================================
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

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
import matplotlib.pyplot as plt
import numpy as np
import winsound

from tensorflow import keras
from tensorflow_core.python.keras import optimizers, losses, metrics, Input, Model
from tensorflow_core.python.keras.layers import Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten
from tensorflow_core.python.keras.layers import Dense, Dropout, BatchNormalization, concatenate, Activation
from tensorflow_core.python.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, LSTM
from keras_preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.regularizers import l2
from tensorflow_core.python.keras.utils import plot_model


# ----------------------------------------------------------------------
# 输出训练的结果
def output_result(results, predictions, y_test):
    print("模型预测-->", end='')
    print("损失值 = {0}，精确度 = {1}".format(results[0], results[1]))
    if label_name == 'age':
        np_arg_max = np.argmax(predictions, 1)
        # print("前 30 个真实的预测数据 =", np.array(X_test[:30], dtype = int))
        print("前 30 个真实的目标数据 =", np.array(y_test[:30], dtype=int))
        print("前 30 个预测的目标数据 =", np.array(np.argmax(predictions[:30], 1), dtype=int))
        print("前 30 个预测的结果数据 =", )
        print(predictions[:30])
        for i in range(10):
            print("类别 {0} 的真实数目：{1}，预测数目：{2}".format(i, sum(y_test == i), sum(np_arg_max == i)))
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


def construct_keras_model(model_type, embedding_weights):
    keras_model = keras.Sequential()
    keras_model.add(Embedding(creative_id_window, embedding_size, input_length=max_len, weights=[embedding_weights], trainable=False))
    if model_type == 'MLP':
        keras_model.add(Flatten())
    elif model_type == 'GM':
        keras_model.add(GlobalMaxPooling1D())
    elif model_type == 'GA':
        keras_model.add(GlobalAveragePooling1D())
    elif model_type == 'Conv1D':
        keras_model.add(Conv1D(64, 2))
        keras_model.add(MaxPooling1D())
        keras_model.add(Conv1D(64,2))
        keras_model.add(MaxPooling1D())
        keras_model.add(Flatten())
    else:
        raise Exception("错误的网络模型类型")

    # keras_model.add(Dropout(0.5))
    # keras_model.add(BatchNormalization())
    # keras_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    keras_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    # keras_model.add(Dropout(0.5))
    # keras_model.add(BatchNormalization())
    keras_model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
    keras_model.summary()
    # print("保存模型的原始结构：", keras_model.save('save_model/word2vec/{0}_m0_{1}.h5'.format(model_type, label_name)))
    keras_model.compile(optimizer=optimizers.RMSprop(lr=RMSProp_lr), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    return keras_model


def construct_keras_api_model(embedding_weights):
    # input_no_time_no_repeat = Input(shape=max_len, dtype='int32')
    # embedded_no_time_no_repeat = Embedding(
    #     creative_id_window,embedding_size,weights=[embedding_weights],trainable=False
    # )(input_no_time_no_repeat)
    # ==================================================================================
    Input_fix_creative_id = Input(
        shape=(math.ceil(time_id_max / period_days) * period_length), dtype='int32', name='input_fix_creative_id'
    )
    Embedded_fix_creative_id = Embedding(
        creative_id_window, embedding_size, weights=[embedding_weights], trainable=False
    )(Input_fix_creative_id)
    # ==================================================================================
    # input_no_time_with_repeat = Input(shape=max_len, dtype='int32')
    # embedded_no_time_with_repeat = Embedding(creative_id_window,embedding_size,weights=[embedding_weights],trainable=False)(input_no_time_with_repeat)

    # ----------------------------------------------------------------------
    GM_x = keras.layers.GlobalMaxPooling1D()(Embedded_fix_creative_id)
    GM_x = Dropout(0.5)(GM_x)
    GM_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001))(GM_x)
    GM_x = BatchNormalization()(GM_x)
    GM_x = Activation('relu')(GM_x)
    GM_x = Dropout(0.5)(GM_x)
    GM_x = Dense(embedding_size // 4, kernel_regularizer=l2(0.001))(GM_x)
    GM_x = BatchNormalization()(GM_x)
    GM_x = Activation('relu')(GM_x)
    GM_x = Dense(1, 'sigmoid')(GM_x)

    # ----------------------------------------------------------------------
    GA_x = GlobalAveragePooling1D()(Embedded_fix_creative_id)
    GA_x = Dropout(0.5)(GA_x)
    GA_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001))(GA_x)
    GA_x = BatchNormalization()(GA_x)
    GA_x = Activation('relu')(GA_x)
    GA_x = Dropout(0.5)(GA_x)
    GA_x = Dense(embedding_size // 4, kernel_regularizer=l2(0.001))(GA_x)
    GA_x = BatchNormalization()(GA_x)
    GA_x = Activation('relu')(GA_x)
    GA_x = Dense(1, 'sigmoid')(GA_x)

    # ==================================================================================
    Conv_creative_id = Conv1D(embedding_size, 15, 5, activation='relu')(Embedded_fix_creative_id)
    # ----------------------------------------------------------------------
    Conv_GM_x = MaxPooling1D(7)(Conv_creative_id)
    Conv_GM_x = Conv1D(embedding_size, 2, 1, activation='relu')(Conv_GM_x)
    Conv_GM_x = GlobalMaxPooling1D()(Conv_GM_x)
    Conv_GM_x = Dropout(0.5)(Conv_GM_x)
    Conv_GM_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001))(Conv_GM_x)
    Conv_GM_x = BatchNormalization()(Conv_GM_x)
    Conv_GM_x = Activation('relu')(Conv_GM_x)
    Conv_GM_x = Dropout(0.5)(Conv_GM_x)
    Conv_GM_x = Dense(embedding_size // 4, kernel_regularizer=l2(0.001))(Conv_GM_x)
    Conv_GM_x = BatchNormalization()(Conv_GM_x)
    Conv_GM_x = Activation('relu')(Conv_GM_x)
    Conv_GM_x = Dense(1, 'sigmoid')(Conv_GM_x)

    # ----------------------------------------------------------------------
    Conv_GA_x = AveragePooling1D(7)(Conv_creative_id)
    Conv_GA_x = Conv1D(embedding_size, 2, 1, activation='relu')(Conv_GA_x)
    Conv_GA_x = GlobalAveragePooling1D()(Conv_GA_x)
    Conv_GA_x = Dropout(0.5)(Conv_GA_x)
    Conv_GA_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001))(Conv_GA_x)
    Conv_GA_x = BatchNormalization()(Conv_GA_x)
    Conv_GA_x = Activation('relu')(Conv_GA_x)
    Conv_GA_x = Dropout(0.5)(Conv_GA_x)
    Conv_GA_x = Dense(embedding_size // 4, kernel_regularizer=l2(0.001))(Conv_GA_x)
    Conv_GA_x = BatchNormalization()(Conv_GA_x)
    Conv_GA_x = Activation('relu')(Conv_GA_x)
    Conv_GA_x = Dense(1, 'sigmoid')(Conv_GA_x)

    # ----------------------------------------------------------------------
    LSTM_x = Conv1D(embedding_size, 14, 7, activation='relu')(Conv_creative_id)
    LSTM_x = LSTM(embedding_size, return_sequences=True)(LSTM_x)
    LSTM_x = LSTM(embedding_size, return_sequences=True)(LSTM_x)
    LSTM_x = LSTM(embedding_size)(LSTM_x)
    LSTM_x = Dropout(0.5)(LSTM_x)
    LSTM_x = Dense(embedding_size // 2, kernel_regularizer=l2(0.001))(LSTM_x)
    LSTM_x = BatchNormalization()(LSTM_x)
    LSTM_x = Activation('relu')(LSTM_x)
    LSTM_x = Dropout(0.5)(LSTM_x)
    LSTM_x = Dense(embedding_size // 4, kernel_regularizer=l2(0.001))(LSTM_x)
    LSTM_x = BatchNormalization()(LSTM_x)
    LSTM_x = Activation('relu')(LSTM_x)
    LSTM_x = Dense(1, 'sigmoid')(LSTM_x)

    # ----------------------------------------------------------------------
    concatenated = concatenate([
        GM_x,
        GA_x,
        Conv_GM_x,
        Conv_GA_x,
        LSTM_x,
    ], axis=-1)
    output_tensor = Dense(1, 'sigmoid')(concatenated)

    keras_api_model = Model(
        [
            # input_no_time_no_repeat,
            Input_fix_creative_id,
            # input_no_time_with_repeat,
        ],
        output_tensor
    )
    keras_api_model.summary()
    plot_model(keras_api_model, to_file='save_model/keras_api_word2vec_model.png')
    print('-' * 5 + ' ' * 3 + "编译模型" + ' ' * 3 + '-' * 5)
    keras_api_model.compile(
        optimizer=optimizers.RMSprop(lr=RMSProp_lr), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy]
    )
    return keras_api_model


def load_word2vec_weights(path):
    from gensim.models import KeyedVectors

    file_name = path + 'creative_id_{0}_{1}_{2}.kv'.format(embedding_size, embedding_window, creative_id_window)
    print('-' * 5 + ' ' * 3 + "加载 word2vec 模型 {0}".format(file_name) + ' ' * 3 + '-' * 5)
    word2vec = KeyedVectors.load(file_name)
    embedding_weights = np.zeros((creative_id_window, embedding_size))
    for word, index in word2vec.vocab.items():
        try:
            embedding_weights[ord(word), :] = word2vec[word]
        except KeyError:
            pass
    pass
    embedding_weights[0, :] = np.zeros(embedding_size)
    print("Word2Vec 模型加载完成。")
    return embedding_weights


def load_data(path, file_infix):
    # 加载数据
    print('-' * 5 + ' ' * 3 + "加载数据集 {0}".format(path) + ' ' * 3 + '-' * 5)
    x_train = np.load(path + 'x_train_' + file_infix + label_name + '.npy', allow_pickle=True)[:, 0]
    y_train = np.load(path + 'y_train_' + file_infix + label_name + '.npy', allow_pickle=True)
    x_test = np.load(path + 'x_test_' + file_infix + label_name + '.npy', allow_pickle=True)[:, 0]
    y_test = np.load(path + 'y_test_' + file_infix + label_name + '.npy', allow_pickle=True)
    print("数据集加载完成。")
    return x_train, y_train, x_test, y_test


def train_keras_model(model):
    # path = 'fix_{0}_{1}_{2}/'.format(period_days, period_length, creative_id_window)
    # file_infix = 'fix_'
    path = 'no_time_no_repeat/'
    file_infix = ''
    X_train, y_train, X_test, y_test = load_data('save_data/' + path, file_infix)
    X_train = pad_sequences(X_train, maxlen=max_len, padding='pre')
    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
    X_test = pad_sequences(X_test, maxlen=max_len, padding='pre')
    results = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test).squeeze()
    output_result(results, predictions, y_test)

    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    model.fit(X_train, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    results = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test).squeeze()
    output_result(results, predictions, y_test)

    pass


def train_keras_api_model(model):
    # 加载数据
    print('-' * 5 + ' ' * 3 + "加载数据集" + ' ' * 3 + '-' * 5)
    # X_train_no_time_no_repeat = pad_sequences(
    #     np.load('save_data/no_time_no_repeat/x_train_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)[:, 0],
    #     maxlen=max_len,
    #     padding='pre'
    # )
    # X_train_no_time_with_repeat = pad_sequences(
    #     np.load('save_data/no_time_with_repeat/x_train_no_time_with_repeat_' + label_name + '.npy', allow_pickle=True)[:,
    #     0],
    #     maxlen=max_len,
    #     padding='pre'
    # )
    X_train_fix_creative_id = np.load(
        'save_data/fix_{0}_{1}_{2}/x_train_fix_{3}.npy'.format(period_days, period_length, creative_id_window, label_name),
        allow_pickle=True
    )[:, 0]
    y_train = np.load('save_data/no_time_no_repeat/y_train_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)
    # X_test_no_time_no_repeat = pad_sequences(
    #     np.load('save_data/no_time_no_repeat/x_test_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)[:, 0],
    #     maxlen=max_len,
    #     padding='pre'
    # )
    # X_test_no_time_with_repeat = pad_sequences(
    #     np.load('save_data/no_time_with_repeat/x_test_no_time_with_repeat_' + label_name + '.npy', allow_pickle=True)[:, 0],
    #     maxlen=max_len,
    #     padding='pre'
    # )
    X_test_fix_creative_id = np.load(
        'save_data/fix_{0}_{1}_{2}/x_test_fix_{3}.npy'.format(period_days, period_length, creative_id_window, label_name),
        allow_pickle=True
    )[:, 0]
    y_test = np.load('save_data/no_time_no_repeat/y_test_no_time_no_repeat_' + label_name + '.npy', allow_pickle=True)
    # ----------------------------------------------------------------------
    # 训练网络模型
    # 使用验证集
    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    history = model.fit(
        {
            # 'no_time_no_repeat': X_train_no_time_no_repeat,
            # 'no_time_with_repeat': X_train_no_time_with_repeat,
            'input_fix_creative_id': X_train_fix_creative_id,
        },
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        use_multiprocessing=True,
        verbose=2
    )
    # import pickle
    # print("保存第一次模型训练的权重", model.save_weights('save_model/word2vec/word2vec_m1_' + label_name + '.bin'))
    # f = open('save_model/word2vec/word2vec_m1_' + label_name + '.pkl', 'wb')
    # pickle.dump(history.history, f)
    # f.close()
    results = model.evaluate(
        {
            # 'no_time_no_repeat': X_test_no_time_no_repeat,
            # 'no_time_with_repeat': X_train_no_time_with_repeat,
            'input_fix_creative_id': X_test_fix_creative_id,
        },
        y_test,
        verbose=0
    )
    predictions = model.predict({
        # 'no_time_no_repeat': X_test_no_time_no_repeat,
        # 'no_time_with_repeat': X_test_no_time_with_repeat,
        'input_fix_creative_id': X_test_fix_creative_id,
    }).squeeze()
    output_result('', '', '')
    # ----------------------------------------------------------------------
    # 不使用验证集，训练次数减半
    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    history = model.fit(
        {
            # 'no_time_no_repeat': X_train_no_time_no_repeat,
            # 'no_time_with_repeat': X_train_no_time_with_repeat,
            'input_fix_creative_id': X_train_fix_creative_id,
        },
        y_train,
        epochs=epochs // 2,
        batch_size=batch_size,
        use_multiprocessing=True,
        verbose=2
    )
    # print("保存第二次模型训练的权重", model.save_weights('save_model/word2vec/word2vec_m2' + label_name + '.bin'))
    # f = open('save_model/word2vec/word2vec_m2' + label_name + '.pkl', 'wb')
    # pickle.dump(history.history, f)
    # f.close()
    results = model.evaluate(
        {
            # 'no_time_no_repeat': X_test_no_time_no_repeat,
            # 'no_time_with_repeat': X_test_no_time_with_repeat,
            'input_fix_creative_id': X_test_fix_creative_id,
        },
        y_test,
        verbose=0
    )
    predictions = model.predict({
        # 'no_time_no_repeat': X_test_no_time_no_repeat,
        # 'no_time_with_repeat': X_test_no_time_with_repeat,
        'input_fix_creative_id': X_test_fix_creative_id,
    }).squeeze()
    output_result('', '', '')


# =====================================================
if __name__ == '__main__':
    print('>' * 15 + ' ' * 3 + "train_single_gender" + ' ' * 3 + '<' * 15)
    # 定义全局定制变量
    RMSProp_lr = 2e-04
    batch_size = 1024
    embedding_size = 32
    embedding_window = 5
    epochs = 20
    label_name = 'gender'
    max_len = 128  # 64:803109，128:882952 个用户；64：1983350，128：2329077 个素材
    period_days = 3
    period_length = 15  # 每个周期的素材数目
    time_id_max = 91
    # 定制 素材库大小 = creative_id_end - creative_id_start = creative_id_num = creative_id_step_size * (1 + 3 + 1)
    creative_id_step_size = 128000
    creative_id_window = creative_id_step_size * 5
    # ----------------------------------------------------------------------
    no_interval_path = 'save_model/word2vec/no_interval/'
    # train_keras_api_model(construct_keras_api_model(load_word2vec_weights(no_interval_path)))
    train_keras_model(construct_keras_model('Conv1D', load_word2vec_weights(no_interval_path)))
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()

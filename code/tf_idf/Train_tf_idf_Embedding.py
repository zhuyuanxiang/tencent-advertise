# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   Train-Single-Input-Model.py
@Version    :   v0.1
@Time       :   2020-07-07 9:04
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   训练单个输入模型
@理解：
"""
# common imports
import pickle

import numpy as np
import winsound

from sklearn.model_selection import train_test_split
from keras import losses
from keras import metrics
from keras import optimizers
from keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Embedding,
    Flatten,
    GlobalMaxPooling1D,
    LSTM,
    GRU, Dropout,
)
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras.regularizers import l2

# ----------------------------------------------------------------------
from generate_data import generate_data_no_interval_with_repeat
from load_data import load_original_data, load_word2vec_weights, load_data_set
from config import creative_id_max, user_id_num, creative_id_step_size, seed
from show_data import show_example_data, show_reslut


# ----------------------------------------------------------------------
# 构建网络模型
def construct_model(label_name):
    model = Sequential()
    # mask_zero 在 MaxPooling 层中不能支持
    model.add(Embedding(creative_id_window, embedding_size, input_length=max_len))
    if model_type == 'MLP':
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(embedding_size * max_len // 4, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
    elif model_type == 'Conv1D':
        model.add(Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(embedding_size * max_len // 4, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
    elif model_type == 'GlobalMaxPooling1D':
        model.add(GlobalMaxPooling1D())
    elif model_type == 'GRU':
        model.add(GRU(embedding_size, dropout=0.2, recurrent_dropout=0.2))
        # model.add(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5))
    elif model_type == 'Conv1D+LSTM':
        model.add(Conv1D(32, 5, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Conv1D(32, 5, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(LSTM(16, dropout=0.5, recurrent_dropout=0.5))
    elif model_type == 'Bidirectional-LSTM':
        model.add(Bidirectional(LSTM(embedding_size, dropout=0.2, recurrent_dropout=0.2)))
    else:
        raise Exception("错误的网络模型类型")

    if label_name == "age":
        model.add(Dense(embedding_size * 10, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(10, activation='softmax', kernel_regularizer=l2(0.001)))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(
            optimizer=optimizers.RMSprop(lr=RMSProp_lr),
            loss=losses.sparse_categorical_crossentropy,
            metrics=[metrics.sparse_categorical_accuracy]
        )
    elif label_name == 'gender':
        model.add(Dense(embedding_size, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(
            optimizer=optimizers.RMSprop(lr=RMSProp_lr),
            loss=losses.binary_crossentropy,
            metrics=[metrics.binary_accuracy])
    else:
        raise Exception("错误的标签类型！")
    return model


def output_parameters():
    print("实验报告参数")
    print("\tuser_id_number =", user_id_num)
    print("\tcreative_id_max =", creative_id_max)
    print("\tcreative_id_step_size =", creative_id_step_size)
    print("\tcreative_id_window =", creative_id_window)
    print("\tcreative_id_begin =", creative_id_begin)
    print("\tcreative_id_end =", creative_id_end)
    print("\tmax_len =", max_len)
    print("\tembedding_size =", embedding_size)
    print("\tepochs =", epochs)
    print("\tbatch_size =", batch_size)
    print("\tRMSProp =", RMSProp_lr)
    pass


# ----------------------------------------------------------------------
# 训练网络模型
def main():
    global embedding_size
    label_name = 'gender'
    no_interval_path = '../../save_model/tf_idf/no_interval/word2vec/'
    data_file_path = '../../save_data/tf_idf/no_interval/with_repeat/'
    model_file_path = '../../save_model/tf_idf/no_interval/with_repeat/'
    file_prefix = 'creative_id_{0}_{1}_{2}_{3}_{4}_'.format(label_name, model_type, max_len, embedding_size, creative_id_window)
    # ----------------------------------------------------------------------
    # 构建模型
    print('=' * 5 + ' ' * 3 + "构建模型" + ' ' * 3 + '=' * 5)
    output_parameters()
    model = construct_model(label_name)
    model.summary()
    print("保存原始模型 → ", end='')
    model.save(model_file_path + file_prefix + 'm0.h5')
    print("模型保存成功！")

    print('=' * 5 + ' ' * 3 + "加载数据" + ' ' * 3 + '=' * 5)
    x_train, y_train, x_test, y_test = load_data_set(data_file_path, label_name)

    data_type = "填充训练数据集"
    print('-' * 5 + ' ' * 3 + data_type + ' ' * 3 + '-' * 5)
    x_train_seq = pad_sequences(x_train, maxlen=max_len, padding='post')
    show_example_data(x_train, y_train, data_type)

    print('=' * 5 + ' ' * 3 + "训练网络" + ' ' * 3 + '=' * 5)
    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    history = model.fit(x_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
    print("保存第一次训练模型 → ", end='')
    model.save_weights(model_file_path + file_prefix + 'm1.bin')
    with open(model_file_path + file_prefix + 'm1.pkl', 'wb') as fname:
        pickle.dump(history.history, fname)
    print("模型保存成功!")

    data_type = "填充测试数据集"
    print('-' * 5 + ' ' * 3 + data_type + ' ' * 3 + '-' * 5)
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
    show_example_data(x_train, y_train, data_type)

    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_reslut(results, predictions, y_test, label_name)

    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    history = model.fit(x_train_seq, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    print("保存第二次训练模型 → ", end='')
    model.save_weights(model_file_path + file_prefix + 'm2.bin')
    with open(model_file_path + file_prefix + 'm2.pkl', 'wb') as fname:
        pickle.dump(history.history, fname)
    print("模型保存成功。")

    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_reslut(results, predictions, y_test, label_name)

    pass


# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 参数说明：
    # model_type = "Bidirectional+LSTM"  # Bidirectional+LSTM：双向 LSTM
    # model_type = "Conv1D"  # Conv1D：1 维卷积神经网络
    # model_type = "Conv1D+LSTM"  # Conv1D+LSTM：1 维卷积神经网络 + LSTM
    # model_type = "GlobalMaxPooling1D"  # GlobalMaxPooling1D：1 维全局池化层
    # model_type = "GlobalMaxPooling1D+MLP"  # GlobalMaxPooling1D+MLP：1 维全局池化层 + 多层感知机
    # model_type = "LSTM"  # LSTM：循环神经网络
    # model_type = "MLP"  # MLP：多层感知机

    # ----------------------------------------------------------------------
    # 定义全局通用变量
    file_name = '../../save_data/tf_idf/train_data_all_tf_idf_v.csv'
    model_type = 'Conv1D'
    RMSProp_lr = 5e-04
    epochs = 10
    batch_size = 256
    # ----------------------------------------------------------------------
    # 定义全局定制变量
    max_len = 128  # 64:803109，128:882952 个用户；64：1983350，128：2329077 个素材
    embedding_size = 32
    creative_id_window = creative_id_step_size * 1
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window
    # 运行训练程序
    main()
    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1000)

# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   Train_sparsity_Word2Vec_MLP.py
@Version    :   v0.1
@Time       :   2020-07-10 12:13
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import numpy as np
import pickle
import winsound
import build_model

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
    GRU,
    Dropout,
    MaxPooling1D
)
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras.regularizers import l2

# ----------------------------------------------------------------------
from load_data import load_word2vec_weights, load_data_set, load_data
from config import creative_id_window, model_file_path, data_file_path
from config import label_name
from config import model_type, RMSProp_lr, epochs, batch_size, max_len, embedding_size, embedding_window
from show_data import show_result, show_parameters

np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)


# ----------------------------------------------------------------------
# 构建网络模型
def construct_model(embedding_weights):
    model = Sequential()
    # mask_zero 在 MaxPooling 层中不能支持
    model.add(Embedding(creative_id_window, embedding_size, input_length=max_len, weights=[embedding_weights], trainable=False))
    if model_type == 'MLP':
        model = build_model.build_mlp(model)
    elif model_type == 'Conv1D+MLP':
        model = build_model.build_conv1d_mlp(model)
    elif model_type == 'Conv1D':
        model = build_model.build_conv1d(model)
    elif model_type == 'LeNet':
        model = build_model.build_conv1d(model)
    elif model_type == 'AlexNet':
        model = build_model.build_conv1d(model)
    elif model_type == 'VGG':
        model = build_model.build_conv1d(model)
    elif model_type == 'NiN':
        model = build_model.build_conv1d(model)
    elif model_type == 'GoogLeNet':
        model = build_model.build_conv1d(model)
    elif model_type == 'ResNet':
        model = build_model.build_conv1d(model)
    elif model_type == 'DenseNet':
        model = build_model.build_conv1d(model)
    elif model_type == 'GlobalMaxPooling1D':
        model = build_model.build_global_max_pooling1d(model)
    elif model_type == 'GRU':
        model = build_model.build_gru(model)
    elif model_type == 'Conv1D+LSTM':
        model = build_model.build_conv1d_lstm(model)
    elif model_type == 'Bidirectional-LSTM':
        model = build_model.build_bidirectional_lstm(model)
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
        model.compile(optimizer=optimizers.RMSprop(lr=RMSProp_lr), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    else:
        raise Exception("错误的标签类型！")
    return model


# ----------------------------------------------------------------------
def main():
    file_prefix = 'embedding_{0}_{1}_'.format(embedding_size, max_len)
    print('-' * 5 + ' ' * 3 + "构建网络模型" + ' ' * 3 + '-' * 5)
    show_parameters()
    model = construct_model(load_word2vec_weights())
    model.summary()
    # print("保存原始模型 →", end='')
    # model.save(model_file_path + file_prefix + 'm0.h5')
    # print("模型保存成功。")

    print('-' * 5 + ' ' * 3 + "加载与填充训练数据集" + ' ' * 3 + '-' * 5)
    # x_train = load_data(data_file_path + 'x_train', '训练数据集')
    # y_train = load_data(data_file_path + 'y_train', '训练数据集')
    x_train = load_data(data_file_path + 'x_train_balance', '平衡的训练数据集')
    y_train = load_data(data_file_path + 'y_train_balance', '平衡的训练数据集')
    x_train_seq = pad_sequences(x_train, maxlen=max_len, padding='post')

    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    history = model.fit(x_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
    # print("保存第一次训练模型 → ", end='')
    # model.save_weights(model_file_path + file_prefix + 'm1.bin')
    # with open(model_file_path + file_prefix + 'm1.pkl', 'wb') as fname:
    #     pickle.dump(history.history, fname)
    # print("模型保存成功。")

    print('-' * 5 + ' ' * 3 + "加载与填充测试数据集" + ' ' * 3 + '-' * 5)
    x_test = load_data(data_file_path + 'x_test', '测试数据集')
    y_test = load_data(data_file_path + 'y_test', '测试数据集')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_result(results, predictions, y_test)

    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    history = model.fit(x_train_seq, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    # print("保存第二次训练模型 → ", end='')
    # model.save_weights(model_file_path + file_prefix + 'm2.bin')
    # with open(model_file_path + file_prefix + 'm2.pkl', 'wb') as fname:
    #     pickle.dump(history.history, fname)
    # print("模型保存成功。")

    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_result(results, predictions, y_test)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
    # 运行结束的提醒
    winsound.Beep(600, 500)
    winsound.Beep(900, 1000)
    winsound.Beep(600, 500)

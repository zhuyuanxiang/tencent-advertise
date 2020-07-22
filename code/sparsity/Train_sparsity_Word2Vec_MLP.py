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

from keras_preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------------
from load_data import load_data
from config import model_file_path, data_file_path, model_file_prefix, save_model, x_train_file_name, y_train_file_name, train_data_type
from config import x_test_file_name, y_test_file_name
from config import model_type, epochs, batch_size, max_len
from show_data import show_result, show_parameters

np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)


# ----------------------------------------------------------------------
# 构建网络模型
def construct_model():
    if model_type == 'MLP':
        model = build_model.build_mlp()
    elif model_type == 'Conv1D+MLP':
        model = build_model.build_conv1d_mlp()
    elif model_type == 'Conv1D':
        model = build_model.build_conv1d()
    elif model_type == 'LeNet':
        model = build_model.build_le_net()
    elif model_type == 'AlexNet':
        model = build_model.build_alex_net()
    elif model_type == 'VGG':
        model = build_model.build_vgg()
    elif model_type == 'NiN':
        model = build_model.build_conv1d()
    elif model_type == 'GoogLeNet':
        model = build_model.build_conv1d()
    elif model_type == 'ResNet':
        model = build_model.build_conv1d()
    elif model_type == 'DenseNet':
        model = build_model.build_conv1d()
    elif model_type == 'GlobalMaxPooling1D':
        model = build_model.build_global_max_pooling1d()
    elif model_type == 'GRU':
        model = build_model.build_gru()
    elif model_type == 'Conv1D+LSTM':
        model = build_model.build_conv1d_lstm()
    elif model_type == 'Bidirectional-LSTM':
        model = build_model.build_bidirectional_lstm()
    else:
        raise Exception("错误的网络模型类型")
    return model


# ----------------------------------------------------------------------
def main():
    print('-' * 5 + ' ' * 3 + "构建网络模型" + ' ' * 3 + '-' * 5)
    show_parameters()
    model = construct_model()
    model.summary()
    save_model_m0(model)

    print('-' * 5 + ' ' * 3 + "加载与填充训练数据集" + ' ' * 3 + '-' * 5)
    x_train = load_data(data_file_path + x_train_file_name, train_data_type)
    y_train = load_data(data_file_path + y_train_file_name, train_data_type)
    x_train_seq = pad_sequences(x_train, maxlen=max_len, padding='post')

    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    history = model.fit(x_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
    save_model_m1(history, model)

    print('-' * 5 + ' ' * 3 + "加载与填充测试数据集" + ' ' * 3 + '-' * 5)
    x_test = load_data(data_file_path + x_test_file_name, '测试数据集')
    y_test = load_data(data_file_path + y_test_file_name, '测试数据集')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_result(results, predictions, y_test)

    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    history = model.fit(x_train_seq, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    save_model_m2(history, model)

    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_result(results, predictions, y_test)


def save_model_m0(model):
    if save_model:
        file_name = model_file_path + model_file_prefix + 'm0.h5'
        print("保存原始模型:{} →".format(file_name), end='')
        model.save(file_name)
        print("模型保存成功。")


def save_model_m2(history, model):
    if save_model:
        print("保存第二次训练模型 → ", end='')
        file_name = model_file_path + model_file_prefix + 'm2.bin'
        model.save_weights(file_name)
        with open(model_file_path + model_file_prefix + 'm2.pkl', 'wb') as fname:
            pickle.dump(history.history, fname)
        print("模型保存成功。")


def save_model_m1(history, model):
    if save_model:
        file_name = model_file_path + model_file_prefix + 'm1.bin'
        print("保存第一次训练模型:{} → ".format(file_name), end='')
        model.save_weights(file_name)
        with open(model_file_path + model_file_prefix + 'm1.pkl', 'wb') as fname:
            pickle.dump(history.history, fname)
        print("模型保存成功。")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
    # 运行结束的提醒
    winsound.Beep(600, 500)
    winsound.Beep(900, 1000)
    winsound.Beep(600, 500)

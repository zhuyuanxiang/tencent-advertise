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
@Desc       :   训练模块
@理解：
"""
import pickle

import config
from build_model_dense_net import build_dense_net
from config import batch_size, epochs, max_len, model_type
from config import data_file_path
from config import model_file_path, model_file_prefix, save_model
from config import x_test_file_name, y_test_file_name
from config import x_train_file_name, y_train_file_name, train_data_type
from tools import beep_end, show_title


# ----------------------------------------------------------------------
# 构建网络模型
def construct_model():
    if model_type == 'MLP':
        from builld_model_MLP import build_mlp
        model = build_mlp()
    elif model_type == 'Conv1D+MLP':
        from build_model_CNN import build_conv1d_mlp
        model = build_conv1d_mlp()
    elif model_type == 'Conv1D':
        from build_model_CNN import build_conv1d
        model = build_conv1d()
    elif model_type == 'LeNet':
        from build_model_CNN import build_le_net
        model = build_le_net()
    elif model_type == 'AlexNet':
        from build_model_CNN import build_alex_net
        model = build_alex_net()
    elif model_type == 'VGG':
        from build_model_vgg import build_vgg
        model = build_vgg()
    elif model_type == 'NiN':
        from build_model_nin import build_nin
        model = build_nin()
    elif model_type == 'GoogLeNet':
        from build_model_google_net import build_google_net
        model = build_google_net()
    elif model_type == 'ResNet':
        from build_model_res_net import build_res_net
        model = build_res_net()
    elif model_type == 'DenseNet':
        model = build_dense_net()
    elif model_type == 'GM':  # GlobalMaxPooling1D
        from build_model_CNN import build_global_max_pooling1d
        model = build_global_max_pooling1d()
    elif model_type == 'GRU':
        from build_model_RNN import build_gru
        model = build_gru()
    elif model_type == 'Conv1D+LSTM':
        from build_model_RNN import build_conv1d_lstm
        model = build_conv1d_lstm()
    elif model_type == 'Bidirectional-LSTM':
        from build_model_RNN import build_bidirectional_lstm
        model = build_bidirectional_lstm()
    else:
        raise Exception("错误的网络模型类型")
    return model


# ----------------------------------------------------------------------
def main():
    from keras_preprocessing.sequence import pad_sequences
    from load_data import load_data
    from show_data import show_result, show_parameters

    show_title("构建网络模型")
    show_parameters()
    model = construct_model()
    model.summary()
    save_model_m0(model)

    show_title("加载与填充{}".format(train_data_type))

    x_train = load_data(data_file_path + x_train_file_name, train_data_type + "(x_train)")[:, 0]
    y_train = load_data(data_file_path + y_train_file_name, train_data_type + "(y_train")
    x_train_seq = pad_sequences(x_train, maxlen=max_len, padding='post')

    x_train_val = load_data(data_file_path + config.x_train_val_file_name, train_data_type + "(" + config.x_train_val_file_name + ")")[:, 0]
    y_train_val = load_data(data_file_path + config.y_train_val_file_name, train_data_type + "(" + config.y_train_val_file_name + ")")
    x_train_val_seq = pad_sequences(x_train_val, maxlen=max_len, padding='post')

    x_val = load_data(data_file_path + config.x_val_file_name, train_data_type + "(" + config.x_val_file_name + ")")[:, 0]
    y_val = load_data(data_file_path + config.y_val_file_name, train_data_type + "(" + config.y_val_file_name + ")")
    x_val_seq = pad_sequences(x_val, maxlen=max_len, padding='post')

    show_title("存在验证集训练网络模型")
    # history = model.fit({'creative_id': x_train_seq}, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
    history = model.fit(x={'creative_id': x_train_val_seq}, y=y_train_val, epochs=epochs, batch_size=batch_size, validation_data=(x_val_seq, y_val), verbose=2)
    save_model_m1(history, model)

    show_title("加载与填充测试数据集")
    x_test = load_data(data_file_path + x_test_file_name, '测试数据集(x_test)')[:, 0]
    y_test = load_data(data_file_path + y_test_file_name, '测试数据集(y_test)')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
    results = model.evaluate({'creative_id': x_test}, y_test, verbose=0)
    predictions = model.predict({'creative_id': x_test}).squeeze()
    show_result(results, predictions, y_test)

    show_title("没有验证集训练网络模型，训练次数减半")
    history = model.fit({'creative_id': x_train_seq}, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    save_model_m2(history, model)

    results = model.evaluate({'creative_id': x_test}, y_test, verbose=0)
    predictions = model.predict({'creative_id': x_test}).squeeze()
    show_result(results, predictions, y_test)


def save_model_m0(model):
    if save_model:
        file_name = model_file_path + model_file_prefix + 'm0.h5'
        print("保存原始模型:{} →".format(file_name), end='')
        model.save(file_name)
        print("模型保存成功。")


def save_model_m1(history, model):
    if save_model:
        file_name = model_file_path + model_file_prefix + 'm1.bin'
        print("保存第一次训练模型:{} → ".format(file_name), end='')
        model.save_weights(file_name)
        with open(model_file_path + model_file_prefix + 'm1.pkl', 'wb') as fname:
            pickle.dump(history.history, fname)
        print("模型保存成功。")


def save_model_m2(history, model):
    if save_model:
        file_name = model_file_path + model_file_prefix + 'm2.bin'
        print("保存第二次训练模型:{} → ".format(file_name), end='')
        model.save_weights(file_name)
        with open(model_file_path + model_file_prefix + 'm2.pkl', 'wb') as fname:
            pickle.dump(history.history, fname)
        print("模型保存成功。")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
    beep_end()

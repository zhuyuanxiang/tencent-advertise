# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   train_day_sequence.py
@Version    :   v0.1
@Time       :   2020-08-20 9:23
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# ----------------------------------------------------------------------
import gc

import numpy as np
from keras import Input
from keras.layers import Dense, Dropout

from config import batch_size
from config import embedding_size
from config import epochs
from config import time_id_max
from config import train_data_type
from src.model.build_model import build_single_model_api
from src.model.build_model import build_single_output_api
from src.model.build_model_res_net import build_residual, build_residual_bn
from tools import beep_end
from tools import show_title


def build_inception(incep_num, x_input):
    x_output = build_cnn_inception(incep_num, x_input)
    # x_output = Dropout(0.2, name=f"Inception_Drop_{incep_num}")(x_output)
    # x_output = build_pool_inception(x_output)
    x_output = build_rnn_inception(incep_num, x_output)
    return x_output


def build_pool_inception(x_output):
    from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
    x_output = build_residual(x_output, embedding_size * 6, 4)
    x_output = Dropout(0.2)(x_output)
    x_output = concatenate([
            GlobalMaxPooling1D()(x_output),
            GlobalAveragePooling1D()(x_output)
    ], axis=-1)
    return x_output


def build_rnn_inception(incep_num, x_output):
    from keras.layers import LSTM, Bidirectional
    x_output = Bidirectional(
            LSTM(embedding_size * 6, dropout=0.2, recurrent_dropout=0.2), name=f"BiLSTM_{incep_num}")(x_output)
    x_output = Dropout(0.2, name=f"RNN_Drop_{incep_num}")(x_output)
    from keras.regularizers import l1
    # from keras.regularizers import l2
    x_output = Dense(
            embedding_size, activation='relu', kernel_regularizer=l1(0.001), name=f"RNN_Dense_{incep_num}")(x_output)
    return x_output


def build_cnn_inception(incep_num, x_input):
    x_output = build_residual_bn(x_input, embedding_size * 3, incep_num)
    from keras.layers import MaxPooling1D
    x_output = MaxPooling1D(7, name=f"CNN_MaxPooling_{incep_num}")(x_output)
    return x_output


def construct_model_single_input():
    Input_creative_id = Input(shape=(time_id_max, embedding_size), name='creative_id')
    x_output = build_inception(1, Input_creative_id)
    return build_single_model_api([Input_creative_id], build_single_output_api(x_output))


def train_single_input():
    from src.data.show_data import show_result, show_parameters

    show_title("构建网络模型")
    show_parameters()
    model = construct_model_single_input()
    model.summary()

    from src.model.save_model import save_model_m0
    save_model_m0(model)

    from src.data.load_data import load_train_val_data
    x_train_val, y_train_val = load_train_val_data()
    from config import day_feature_idx
    x_train_val = single_data_reshape(day_feature_idx, x_train_val, y_train_val.shape[0])
    from src.data.load_data import load_val_data
    x_val, y_val = load_val_data()
    x_val = single_data_reshape(day_feature_idx, x_val, y_val.shape[0])
    show_title("存在验证集训练网络模型")
    history = model.fit(x_train_val, y_train_val, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), verbose=2)
    del x_train_val, x_val, y_train_val, y_val
    gc.collect()

    from src.model.save_model import save_model_m1
    save_model_m1(history, model)

    from src.data.load_data import load_test_data
    show_title("加载与填充测试数据集")
    x_test, y_test = load_test_data()
    x_test = single_data_reshape(day_feature_idx, x_test, y_test.shape[0])
    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_result(results, predictions, y_test)

    show_title("没有验证集训练网络模型，训练次数减半")
    from src.data.load_data import load_train_data
    show_title("加载与填充{}".format(train_data_type))
    x_train, y_train = load_train_data()
    x_train = single_data_reshape(day_feature_idx, x_train, y_train.shape[0])

    history = model.fit(x_train, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    from src.model.save_model import save_model_m2
    save_model_m2(history, model)

    results = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test).squeeze()
    show_result(results, predictions, y_test)
    pass


def single_data_reshape(feature_idx, x_data, data_size):
    # return x_data.reshape([data_size, time_id_max, 3 * embedding_size])
    return x_data[:, :, feature_idx, :].reshape([data_size, time_id_max, embedding_size])


def construct_model_multi_input():
    Input_creative_id_min = Input(shape=(time_id_max, embedding_size), name='creative_id_min')
    x_creative_id_min = build_inception(1, Input_creative_id_min)
    Input_creative_id_max = Input(shape=(time_id_max, embedding_size), name='creative_id_max')
    x_creative_id_max = build_inception(2, Input_creative_id_max)
    Input_creative_id_mean = Input(shape=(time_id_max, embedding_size), name='creative_id_mean')
    x_creative_id_mean = build_inception(3, Input_creative_id_mean)
    from keras.layers import concatenate
    x_output = concatenate([x_creative_id_min, x_creative_id_max, x_creative_id_mean], axis=-1)
    x_output = Dropout(0.2)(x_output)
    return build_single_model_api([Input_creative_id_min, Input_creative_id_max, Input_creative_id_mean],
                                  build_single_output_api(x_output))


def train_multi_input():
    from src.data.show_data import show_result, show_parameters

    show_title("构建网络模型")
    show_parameters()
    model = construct_model_multi_input()
    model.summary()

    from src.model.save_model import save_model_m0
    show_title("保存网络模型")
    save_model_m0(model)

    from src.data.load_data import load_train_val_data
    x_train_val, y_train_val = load_train_val_data()
    x_train_val = reshape_data(x_train_val, y_train_val.shape[0])
    from src.data.load_data import load_val_data
    x_val, y_val = load_val_data()
    x_val = reshape_data(x_val, y_val.shape[0])
    show_title("存在验证集训练网络模型")
    history = model.fit(x={
            'creative_id_min': x_train_val[0],
            'creative_id_max': x_train_val[1],
            'creative_id_mean': x_train_val[2],
    }, y=y_train_val, epochs=epochs, batch_size=batch_size,
            validation_data=(
                    {
                            'creative_id_min': x_val[0],
                            'creative_id_max': x_val[1],
                            'creative_id_mean': x_val[2],
                    }, y_val), verbose=2)
    del x_train_val, x_val, y_train_val, y_val
    gc.collect()

    from src.model.save_model import save_model_m1
    save_model_m1(history, model)

    from src.data.load_data import load_test_data
    show_title("加载与填充测试数据集")
    x_test, y_test = load_test_data()
    x_test = reshape_data(x_test, y_test.shape[0])
    results = model.evaluate({
            'creative_id_min': x_test[0],
            'creative_id_max': x_test[1],
            'creative_id_mean': x_test[2],
    }, y_test, verbose=0)
    predictions = model.predict({
            'creative_id_min': x_test[0],
            'creative_id_max': x_test[1],
            'creative_id_mean': x_test[2],
    }).squeeze()
    show_result(results, predictions, y_test)

    show_title("没有验证集训练网络模型，训练次数减半")
    from src.data.load_data import load_train_data
    show_title("加载与填充{}".format(train_data_type))
    x_train, y_train = load_train_data()
    x_train = reshape_data(x_train, y_train.shape[0])
    # history = model.fit({'creative_id': x_train_seq}, y_train, epochs=epochs, batch_size=batch_size,
    #                     validation_split=0.2, verbose=2)

    history = model.fit({
            'creative_id_min': x_train[0],
            'creative_id_max': x_train[1],
            'creative_id_mean': x_train[2],
    }, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    from src.model.save_model import save_model_m2
    save_model_m2(history, model)

    results = model.evaluate({
            'creative_id_min': x_test[0],
            'creative_id_max': x_test[1],
            'creative_id_mean': x_test[2],
    }, y_test, verbose=0)
    predictions = model.predict({
            'creative_id_min': x_test[0],
            'creative_id_max': x_test[1],
            'creative_id_mean': x_test[2],
    }).squeeze()
    show_result(results, predictions, y_test)
    pass


def reshape_data(x_train_val, data_size):
    result = []
    from config import day_feature_num
    for feature_idx in range(day_feature_num):
        result.append(x_train_val[:, :, feature_idx, :].reshape([data_size, time_id_max, embedding_size]))
    return np.array(result)


if __name__ == '__main__':
    # 运行结束的提醒
    # train_single_input()
    train_multi_input()
    beep_end()

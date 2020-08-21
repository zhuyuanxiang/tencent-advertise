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

from keras import Input
from keras.layers import Dense, Dropout
from keras.regularizers import l2

from config import batch_size
from config import embedding_size
from config import epochs
from config import time_id_max
from config import train_data_type
from src.model.build_model import build_single_model_api
from src.model.build_model import build_single_output_api
from src.model.build_model_res_net import build_residual
from tools import beep_end
from tools import show_title


def construct_model():
    input_creative_id = Input(shape=(time_id_max, embedding_size), name='creative_id')
    model_input = [input_creative_id]
    x_output = build_residual(model_input[0], embedding_size * 3, 3)
    x_output = Dropout(0.2)(x_output)
    from keras.layers import MaxPooling1D
    x_output = MaxPooling1D(7)(x_output)
    # from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
    # x_output = build_residual(x_output, embedding_size * 6, 4)
    # x_output = concatenate([
    #         GlobalMaxPooling1D()(x_output),
    #         GlobalAveragePooling1D()(x_output)
    # ], axis=-1)

    from keras.layers import LSTM, Bidirectional
    x_output = Bidirectional(LSTM(embedding_size * 6, dropout=0.2, recurrent_dropout=0.2))(x_output)
    x_output = Dropout(0.2)(x_output)
    x_output = Dense(embedding_size, activation='relu', kernel_regularizer=l2(0.001))(x_output)

    model_output = build_single_output_api(x_output)
    return build_single_model_api(model_input, model_output)


def main():
    from src.data.show_data import show_result, show_parameters

    show_title("构建网络模型")
    show_parameters()
    model = construct_model()
    model.summary()

    from src.model.save_model import save_model_m0
    show_title("保存网络模型")
    save_model_m0(model)

    from src.data.load_data import load_train_val_data
    x_train_val, y_train_val = load_train_val_data()
    # x_train_val = x_train_val.reshape([y_train_val.shape[0], time_id_max, 3 * embedding_size])
    feature_idx = 2
    x_train_val = x_train_val[:, :, feature_idx, :].reshape([y_train_val.shape[0], time_id_max, embedding_size])
    from src.data.load_data import load_val_data
    x_val, y_val = load_val_data()
    # x_val = x_val.reshape([y_val.shape[0], time_id_max, 3 * embedding_size])
    x_val = x_val[:, :, feature_idx, :].reshape([y_val.shape[0], time_id_max, embedding_size])
    show_title("存在验证集训练网络模型")
    history = model.fit(x={'creative_id': x_train_val}, y=y_train_val, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), verbose=2)
    del x_train_val, x_val, y_train_val, y_val
    gc.collect()

    from src.model.save_model import save_model_m1
    save_model_m1(history, model)

    from src.data.load_data import load_test_data
    show_title("加载与填充测试数据集")
    x_test, y_test = load_test_data()
    # x_test = x_test.reshape([y_test.shape[0], time_id_max, 3 * embedding_size])
    x_test = x_test[:, :, feature_idx, :].reshape([y_test.shape[0], time_id_max, embedding_size])
    results = model.evaluate({'creative_id': x_test}, y_test, verbose=0)
    predictions = model.predict({'creative_id': x_test}).squeeze()
    show_result(results, predictions, y_test)

    show_title("没有验证集训练网络模型，训练次数减半")
    from src.data.load_data import load_train_data
    show_title("加载与填充{}".format(train_data_type))
    x_train, y_train = load_train_data()
    # x_train = x_train.reshape([y_train.shape[0], time_id_max, 3 * embedding_size])
    x_train = x_train[:, :, feature_idx, :].reshape([y_train.shape[0], time_id_max, embedding_size])
    # history = model.fit({'creative_id': x_train_seq}, y_train, epochs=epochs, batch_size=batch_size,
    #                     validation_split=0.2, verbose=2)

    history = model.fit({'creative_id': x_train}, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    from src.model.save_model import save_model_m2
    save_model_m2(history, model)

    results = model.evaluate({'creative_id': x_test}, y_test, verbose=0)
    predictions = model.predict({'creative_id': x_test}).squeeze()
    show_result(results, predictions, y_test)
    pass


if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()

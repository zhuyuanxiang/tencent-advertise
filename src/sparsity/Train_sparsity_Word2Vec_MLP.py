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

from config import batch_size, epochs, max_len, model_type
from config import train_data_type
from src.data.load_data import load_test_data
from src.model.build_model_dense_net import build_dense_net
from src.model.save_model import save_model_m0, save_model_m1, save_model_m2
from tools import beep_end, show_title


# ----------------------------------------------------------------------
# 构建网络模型
def construct_model():
    if model_type == 'MLP':
        from src.model.builld_model_MLP import build_mlp
        model = build_mlp()
    elif model_type == 'Conv1D+MLP':
        from src.model.build_model_CNN import build_conv1d_mlp
        model = build_conv1d_mlp()
    elif model_type == 'Conv1D':
        from src.model.build_model_CNN import build_conv1d
        model = build_conv1d()
    elif model_type == 'LeNet':
        from src.model.build_model_CNN import build_le_net
        model = build_le_net()
    elif model_type == 'AlexNet':
        from src.model.build_model_CNN import build_alex_net
        model = build_alex_net()
    elif model_type == 'VGG':
        from src.model.build_model_vgg import build_vgg
        model = build_vgg()
    elif model_type == 'NiN':
        from src.model.build_model_nin import build_nin
        model = build_nin()
    elif model_type == 'GoogLeNet':
        from src.model.build_model_google_net import build_google_net
        model = build_google_net()
    elif model_type == 'ResNet':
        from src.model.build_model_res_net import build_res_net
        model = build_res_net()
    elif model_type == 'DenseNet':
        model = build_dense_net()
    elif model_type == 'GM':  # GlobalMaxPooling1D
        from src.model.build_model_CNN import build_global_max_pooling1d
        model = build_global_max_pooling1d()
    elif model_type == 'GRU':
        from src.model.build_model_RNN import build_gru
        model = build_gru()
    elif model_type == 'Conv1D+LSTM':
        from src.model.build_model_RNN import build_conv1d_lstm
        model = build_conv1d_lstm()
    elif model_type == 'Bidirectional-LSTM':
        from src.model.build_model_RNN import build_bidirectional_lstm
        model = build_bidirectional_lstm()
    else:
        raise Exception("错误的网络模型类型")
    return model


# ----------------------------------------------------------------------
def main():
    from keras_preprocessing.sequence import pad_sequences
    from src.data.show_data import show_result, show_parameters
    from src.data.load_data import load_train_data, load_train_val_data, load_val_data

    show_title("构建网络模型")
    show_parameters()
    model = construct_model()
    model.summary()
    save_model_m0(model)

    show_title("加载与填充{}".format(train_data_type))

    x_train_val, y_train_val = load_train_val_data()
    x_train_val_seq = pad_sequences(x_train_val, maxlen=max_len, padding='post')

    x_val, y_val = load_val_data()
    x_val_seq = pad_sequences(x_val, maxlen=max_len, padding='post')

    show_title("存在验证集训练网络模型")
    history = model.fit(x={'creative_id': x_train_val_seq}, y=y_train_val, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val_seq, y_val), verbose=2)
    save_model_m1(history, model)

    show_title("加载与填充测试数据集")
    x_test, y_test = load_test_data()
    x_test_seq = pad_sequences(x_test, maxlen=max_len, padding='post')
    results = model.evaluate({'creative_id': x_test_seq}, y_test, verbose=0)
    predictions = model.predict({'creative_id': x_test_seq}).squeeze()
    show_result(results, predictions, y_test)

    show_title("没有验证集训练网络模型，训练次数减半")
    x_train, y_train = load_train_data()
    x_train_seq = pad_sequences(x_train, maxlen=max_len, padding='post')
    # history = model.fit({'creative_id': x_train_seq}, y_train, epochs=epochs, batch_size=batch_size,
    #                     validation_split=0.2, verbose=2)
    history = model.fit({'creative_id': x_train_seq}, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=2)
    save_model_m2(history, model)

    results = model.evaluate({'creative_id': x_test_seq}, y_test, verbose=0)
    predictions = model.predict({'creative_id': x_test_seq}).squeeze()
    show_result(results, predictions, y_test)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
    beep_end()

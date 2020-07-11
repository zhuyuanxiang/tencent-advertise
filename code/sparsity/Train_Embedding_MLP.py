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
from load_data import load_original_data, load_word2vec_weights
from config import creative_id_max, user_id_num, creative_id_step_size, seed
from show_data import show_example_data, show_reslut


# ----------------------------------------------------------------------
# 构建网络模型
def construct_model(label_name):
    output_parameters()
    model = Sequential()
    # mask_zero 在 MaxPooling 层中不能支持
    model.add(Embedding(creative_id_window, embedding_size, input_length=max_len))
    if model_type == 'MLP':
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(embedding_size * max_len // 4, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
    elif model_type == 'Conv1D':
        model.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Conv1D(32, 7, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(GlobalMaxPooling1D())
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
    no_interval_path = '../../save_model/sparsity/no_interval/word2vec/'
    data_file_path = '../../save_data/sparsity/no_interval/with_repeat/'
    model_file_path = '../../save_model/sparsity/no_interval/with_repeat/'
    # ----------------------------------------------------------------------
    # 构建模型
    print('-' * 5 + ' ' * 3 + "构建网络模型" + ' ' * 3 + '-' * 5)
    output_parameters()
    model = construct_model(label_name)
    model.summary()

    # ----------------------------------------------------------------------
    # 加载数据
    field_list = [
        "user_id",  # 0
        "creative_id_inc_sparsity",  # 1
        "time_id",  # 2
        "click_times",  # 3, click_times 属于值，不属于编号，不能再减1
    ]
    label_list = ['age', 'gender']
    x_csv, y_csv = load_original_data(file_name, field_list, label_list)
    show_example_data(x_csv, y_csv)
    # ----------------------------------------------------------------------
    # 清洗数据集，生成所需要的数据
    print('-' * 5 + ' ' * 3 + "清洗数据集" + ' ' * 3 + '-' * 5)
    X_doc, y_doc = generate_data_no_interval_with_repeat(x_csv, y_csv[:, label_list.index(label_name)], creative_id_begin, creative_id_end)
    show_example_data(X_doc, y_doc)
    # ----------------------------------------------------------------------
    # 填充数据集
    x_doc_seq = pad_sequences(X_doc, maxlen=max_len, padding='post')
    # print('-' * 5 + ' ' * 3 + "填充数据集" + ' ' * 3 + '-' * 5)
    # output_example_data(x_doc_seq, y_seq)
    # ----------------------------------------------------------------------
    print('-' * 5 + ' ' * 3 + "拆分数据集" + ' ' * 3 + '-' * 5)
    X_train, X_test, y_train, y_test = train_test_split(x_doc_seq, y_doc, random_state=seed, stratify=y_doc)
    print("训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))

    # ----------------------------------------------------------------------
    # 训练网络模型
    # 使用验证集
    print('-' * 5 + ' ' * 3 + "使用验证集训练网络模型" + ' ' * 3 + '-' * 5)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, use_multiprocessing=True, verbose=2)
    results = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test).squeeze()
    show_reslut(results, predictions, y_test, label_name)

    # ----------------------------------------------------------------------
    # 不使用验证集，训练次数减半
    print('-' * 5 + ' ' * 3 + "不使用验证集训练网络模型，训练次数减半" + ' ' * 3 + '-' * 5)
    history = model.fit(X_train, y_train, epochs=epochs // 2, batch_size=batch_size, use_multiprocessing=True, verbose=2)
    results = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test).squeeze()
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
    file_name = '../../save_data/sparsity/train_data_all_sparsity_v.csv'
    model_type = 'MLP'
    RMSProp_lr = 5e-04
    epochs = 20
    batch_size = 256
    # ----------------------------------------------------------------------
    # 定义全局定制变量
    max_len = 128  # 64:803109，128:882952 个用户；64：1983350，128：2329077 个素材
    embedding_size = 32
    creative_id_window = creative_id_step_size * 5
    creative_id_begin = creative_id_step_size * 0
    creative_id_end = creative_id_begin + creative_id_window
    # 运行训练程序
    print('-' * 5 + ' ' * 3 + "素材数:{0}".format(creative_id_window) + ' ' * 3 + '-' * 5)
    main()
    # train_model(x_csv, y_csv[:, 0], label_name='age')
    # train_model(x_csv, y_csv[:, 1], label_name='gender')
    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1000)

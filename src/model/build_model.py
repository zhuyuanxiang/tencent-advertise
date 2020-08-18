# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model.py
@Version    :   v0.1
@Time       :   2020-07-17 8:32
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   建造各种模型
@理解：
"""
# common imports
from config import creative_id_window, embedding_size, max_len, label_name, model_type, learning_rate
from keras import Sequential, optimizers, losses, metrics, Input, Model
from keras.layers import Embedding, Dense
from keras.regularizers import l2
from src.data.load_data import load_word2vec_weights
from tensorflow import keras


# ----------------------------------------------------------------------
# Single Sequential
def build_single_input():
    model = Sequential(name='creative_id')
    # mask_zero 在 MaxPooling 层中不能支持
    model.add(Embedding(creative_id_window, embedding_size, input_length=max_len, weights=[load_word2vec_weights()], trainable=False))
    return model


def build_single_output(model: keras.Sequential):
    if label_name == "age":
        model.add(Dense(embedding_size * 10, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(10, activation='softmax', kernel_regularizer=l2(0.001)))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(
            optimizer=optimizers.RMSprop(learning_rate),
            loss=losses.sparse_categorical_crossentropy,
            metrics=[metrics.sparse_categorical_accuracy]
        )
    elif label_name == 'gender':
        # model.add(Dense(embedding_size, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
        print("%s——模型构建完成！" % model_type)
        print("* 编译模型")
        model.compile(
            # optimizer=optimizers.RMSprop(lr=RMSProp_lr),
            optimizer=optimizers.Adam(learning_rate),
            loss=losses.binary_crossentropy,
            metrics=[metrics.binary_accuracy]
        )
    else:
        raise Exception("错误的标签类型！")
    return model


# ----------------------------------------------------------------------
# Single Input API
def build_single_model_api(model_input, model_output):
    model = Model(model_input, model_output)
    print("%s——模型构建完成！" % model_type)
    print("* 编译模型")
    if label_name == 'age':
        model.compile(
            optimizer=optimizers.RMSprop(learning_rate),
            loss=losses.sparse_categorical_crossentropy,
            metrics=[metrics.sparse_categorical_accuracy]
        )
    elif label_name == 'gender':
        model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    else:
        raise Exception("错误的标签类型！")
    return model


def build_single_output_api(concatenated):
    if label_name == 'age':
        model_output = Dense(10, activation='softmax', kernel_regularizer=l2(0.001))(concatenated)
    elif label_name == 'gender':
        model_output = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))(concatenated)
    else:
        raise Exception("错误的标签类型！")
    return model_output


# ----------------------------------------------------------------------
# Multi Input API
def build_creative_id_input():
    input_creative_id = Input(shape=(max_len,), dtype='int32', name='creative_id')
    return input_creative_id


def build_product_category_input():
    input_product_category = Input(shape=(max_len,), dtype='int32', name='product_category')
    return input_product_category


def build_multi_input_api():
    embedded_creative_id=build_embedded_creative_id()
    return []


def build_embedded_creative_id(model_input):
    embedded_creative_id = Embedding(
        creative_id_window, embedding_size, input_length=max_len, weights=[load_word2vec_weights()], trainable=False
    )
    x_input = embedded_creative_id(model_input[0])
    return x_input

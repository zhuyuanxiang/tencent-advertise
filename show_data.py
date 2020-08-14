# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
--------------------------------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   show_data.py
@Version    :   v0.1
@Time       :   2020-06-07 15:41
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :   数据展示模块
@理解：
"""
import numpy as np

# ----------------------------------------------------------------------
import config
from config import user_id_max
from config import creative_id_max, creative_id_step_size
from config import max_len, embedding_size, embedding_window
from config import model_type, epochs, batch_size, learning_rate
from config import label_name


def show_example_data(X, y, data_type='原始数据'):
    if config.show_data:
        print(data_type + ":(X[0], y[0]) =", X[0], y[0])
        print(data_type + ":(X[30], y[30]) =", X[30], y[30])
        print(data_type + ":(X[600], y[600]) =", X[600], y[600])
        if len(y) > 8999:
            print(data_type + "：(X[8999], y[8999]) =", X[8999], y[8999])
        if len(y) > 119999:
            print(data_type + "：(X[119999], y[119999]) =", X[119999], y[119999])
        if len(y) > 224999:
            print(data_type + "：(X[224999], y[224999]) =", X[224999], y[224999])
        if len(y) > 674999:
            print(data_type + "：(X[674999], y[674999]) =", X[674999], y[674999])
        if len(y) > 899999:
            print(data_type + "：(X[899999], y[899999]) =", X[899999], y[899999])
            pass
        pass


# ----------------------------------------------------------------------
def show_original_x_data(X, data_type='原始数据'):
    if config.show_data:
        print('-' * 5 + "   展示数据集   " + '-' * 5)
        print(data_type + ":(X[0]) =", X[0])
        print(data_type + ":(X[30]) =", X[30])
        print(data_type + ":(X[600]) =", X[600])
        if len(X) > 8999:
            print(data_type + "：(X[8999]) =", X[8999])
        if len(X) > 119999:
            print(data_type + "：(X[119999]) =", X[119999])
        if len(X) > 224999:
            print(data_type + "：(X[224999]) =", X[224999])
        if len(X) > 674999:
            print(data_type + "：(X[674999]) =", X[674999])
        if len(X) > 899999:
            print(data_type + "：(X[899999]) =", X[899999])
            pass
        pass


# ----------------------------------------------------------------------
def show_data_result(data, data_type='原始数据'):
    if isinstance(data[0], list) and isinstance(data[0][0], str):
        show_word2vec_data(data, data_type)
    else:
        show_original_data(data, data_type)


# ----------------------------------------------------------------------
def show_original_data(data, data_type='原始数据'):
    if config.show_data:
        print('-' * 5 + "   展示数据集   " + '-' * 5)
        print(data_type + ":(X[0]) =", data[0])
        print(data_type + ":(X[30]) =", data[30])
        print(data_type + ":(X[600]) =", data[600])
        if len(data) > 8999:
            print(data_type + "：(X[8999]) =", data[8999])
        if len(data) > 119999:
            print(data_type + "：(X[119999]) =", data[119999])
        if len(data) > 224999:
            print(data_type + "：(X[224999]) =", data[224999])
        if len(data) > 674999:
            print(data_type + "：(X[674999]) =", data[674999])
        if len(data) > 899999:
            print(data_type + "：(X[899999]) =", data[899999])
            pass
        pass


# ----------------------------------------------------------------------
def show_word2vec_data(data, data_type='原始数据'):
    if config.show_data:
        print('-' * 5 + "   展示数据集   " + '-' * 5)
        print(data_type + ":(X[0]) =", data[0])
        print(data_type + ":(X[0]) =", [ord(x) for x in data[0]])
        print(data_type + ":(X[30]) =", [ord(x) for x in data[30]])
        print(data_type + ":(X[600]) =", [ord(x) for x in data[600]])
        if len(data) > 8999:
            print(data_type + "：(X[8999]) =", [ord(x) for x in data[8999]])
        if len(data) > 119999:
            print(data_type + "：(X[119999]) =", [ord(x) for x in data[119999]])
        if len(data) > 224999:
            print(data_type + "：(X[224999]) =", [ord(x) for x in data[224999]])
        if len(data) > 674999:
            print(data_type + "：(X[674999]) =", [ord(x) for x in data[674999]])
        if len(data) > 899999:
            print(data_type + "：(X[899999]) =", [ord(x) for x in data[899999]])
            pass
        pass


# ----------------------------------------------------------------------
# 输出训练的结果
def show_result(results, predictions, y_test):
    if config.show_result:
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


# ----------------------------------------------------------------------
def show_parameters():
    if config.show_parameter:
        print("实验报告参数")
        print('-' * 5 + ' ' * 3 + "与数据相关的参数" + ' ' * 3 + '-' * 5)
        print("\tuser_id_max =", user_id_max)
        print("\tcreative_id_max =", creative_id_max)
        print("\tcreative_id_step_size =", creative_id_step_size)
        print("\tcreative_id_window =", config.creative_id_window)
        print("\tcreative_id_begin =", config.creative_id_begin)
        print("\tcreative_id_end =", config.creative_id_end)
        print("\tmax_len =", max_len)
        print('-' * 5 + ' ' * 3 + "与Word2Vec相关的参数" + ' ' * 3 + '-' * 5)
        print("\tembedding_size =", embedding_size)
        print("\tembedding_window =", embedding_window)
        print('-' * 5 + ' ' * 3 + "与模型相关的参数" + ' ' * 3 + '-' * 5)
        print("\tmodel_type =", model_type)
        print("\tepochs =", epochs)
        print("\tbatch_size =", batch_size)
        print("\tRMSProp =", learning_rate)
        pass

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


# ----------------------------------------------------------------------
def show_example_data(X, y, data_type='原始数据'):
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
def show_word2vec_data(X, data_type='原始数据'):
    print('-' * 5 + "   展示数据集   " + '-' * 5)
    print(data_type + ":(X[0]) =", X[0])
    print(data_type + ":(X[0]) =", [ord(x) for x in X[0]])
    print(data_type + ":(X[30]) =", [ord(x) for x in X[30]])
    print(data_type + ":(X[600]) =", [ord(x) for x in X[600]])
    if len(X) > 8999:
        print(data_type + "：(X[8999]) =", [ord(x) for x in X[8999]])
    if len(X) > 119999:
        print(data_type + "：(X[119999]) =", [ord(x) for x in X[119999]])
    if len(X) > 224999:
        print(data_type + "：(X[224999]) =", [ord(x) for x in X[224999]])
    if len(X) > 674999:
        print(data_type + "：(X[674999]) =", [ord(x) for x in X[674999]])
    if len(X) > 899999:
        print(data_type + "：(X[899999]) =", [ord(x) for x in X[899999]])
        pass
    pass

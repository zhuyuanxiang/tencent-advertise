# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   DL-Age-Keras.py
@Version    :   v0.1
@Time       :   2020-06-11 9:04
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# common imports
import os
import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import winsound
from preprocessing import data_no_sequence, data_sequence, load_data, data_sequence_no_start
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from network import construct_model

# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ >= "1.18.1"


def train_model():
    # ----------------------------------------------------------------------
    # 加载数据
    X_data, y_data = load_data(file_name, label_name)
    # ----------------------------------------------------------------------
    # 清洗数据集
    X_doc, y_doc = data_no_sequence(X_data, y_data, user_id_num, creative_id_num)
    # ----------------------------------------------------------------------
    # 填充数据集
    X = pad_sequences(X_doc, maxlen = max_len, padding = 'post')
    y = y_doc
    # ----------------------------------------------------------------------
    print("* 拆分数据集")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, stratify = y)
    print("\t训练数据集（train_data）：%d 条数据；测试数据集（test_data）：%d 条数据" % ((len(y_train)), (len(y_test))))
    # ----------------------------------------------------------------------
    # 构建模型
    model = construct_model(creative_id_num, embedding_size, max_len, RMSProp_lr, model_type,
                            label_name)

    # ----------------------------------------------------------------------
    def output_result():
        '''
        输出训练的结果
        :return:
        '''
        print("\t模型预测-->", end = '')
        print("\t损失值 = {}，精确度 = {}".format(results[0], results[1]))
        if label_name == 'age':
            print("\t前10个真实的目标数据 =", np.array(y_test[:10], dtype = int))
            print("\t前10个预测的目标数据 =", np.array(np.argmax(predictions[:10], 1), dtype = int))
            print("\t前10个预测的结果数据 =", )
            print(predictions[:10])
        elif label_name == 'gender':
            print("sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) = error% =",
                  sum(abs(np.array(predictions > 0.5, dtype = int) - y_test)) / sum(y_test) * 100,
                  '%')
            print("前100个真实的目标数据 =", np.array(y_test[:100], dtype = int))
            print("前100个预测的目标数据 =", np.array(predictions[:100] > 0.5, dtype = int))
            print("sum(predictions>0.5) =", sum(predictions > 0.5))
            print("sum(y_test) =", sum(y_test))
            print("sum(abs(predictions-y_test))=error_number=",
                  sum(abs(np.array(predictions > 0.5, dtype = int) - y_test)))
        else:
            print("错误的标签名称：", label_name)
            pass
        print("实验报告参数")
        print("user_id_number =", user_id_num)
        print("creative_id_number =", creative_id_num)
        print("max_len =", max_len)
        print("embedding_size =", embedding_size)
        print("epochs =", epochs)
        print("batch_size =", batch_size)
        print("RMSProp =", RMSProp_lr)
        pass

    # ----------------------------------------------------------------------
    print("* 训练模型")
    print("训练数据(X_train[0], y_train[0]) =", X_train[0], y_train[0])
    print("训练数据(X_train[30], y_train[30]) =", X_train[30], y_train[30])
    print("训练数据(X_train[600], y_train[600]) =", X_train[600], y_train[600])
    print("训练数据(X_train[9000], y_train[9000]) =", X_train[9000], y_train[9000])
    # ----------------------------------------------------------------------
    # 使用验证集
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
              validation_split = 0.2, use_multiprocessing = True, verbose = 2)
    results = model.evaluate(X_test, y_test, verbose = 0)
    predictions = model.predict(X_test).squeeze()
    output_result()

    # ----------------------------------------------------------------------
    # 不使用验证集
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
              use_multiprocessing = True, verbose = 2)
    results = model.evaluate(X_test, y_test, verbose = 0)
    predictions = model.predict(X_test).squeeze()
    output_result()


# ----------------------------------------------------------------------
if __name__ == '__main__':
    # model_type = "Bidirectional+LSTM"  # Bidirectional+LSTM：双向 LSTM
    # model_type = "Conv1D"  # Conv1D：1 维卷积神经网络
    # model_type = "Conv1D+LSTM"  # Conv1D+LSTM：1 维卷积神经网络 + LSTM
    # model_type = "GlobalMaxPooling1D"  # GlobalMaxPooling1D：1 维全局池化层
    # model_type = "GlobalMaxPooling1D+MLP"  # GlobalMaxPooling1D+MLP：1 维全局池化层 + 多层感知机
    # model_type = "LSTM"  # LSTM：循环神经网络
    # model_type = "MLP"  # MLP：多层感知机
    # ----------------------------------------------------------------------
    # 定义全局变量
    file_name = './data/train_data_no_sequence.csv'
    user_id_num = 900000  # 用户数
    model_type = "GlobalMaxPooling1D"
    RMSProp_lr = 5e-04
    epochs = 20
    batch_size = 256

    label_name = 'age'
    max_len = 128
    embedding_size = 128
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    max_len = 128
    embedding_size = 256
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    max_len = 256
    embedding_size = 128
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    max_len = 256
    embedding_size = 256
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    label_name = 'gender'
    max_len = 128
    embedding_size = 128
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    max_len = 128
    embedding_size = 256
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    max_len = 256
    embedding_size = 128
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    max_len = 256
    embedding_size = 256
    creative_id_num = 50000  # 素材数
    train_model()
    creative_id_num = 75000
    train_model()
    creative_id_num = 100000
    train_model()
    creative_id_num = 125000
    train_model()
    creative_id_num = 150000
    train_model()
    creative_id_num = 175000
    train_model()
    creative_id_num = 200000
    train_model()

    # 运行结束的提醒
    winsound.Beep(900, 500)
    winsound.Beep(600, 1000)
    plt.show()

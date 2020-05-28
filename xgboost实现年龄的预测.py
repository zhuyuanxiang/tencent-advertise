# -*- encoding: utf-8 -*-
"""
@Author     :
@Contact    :
@site       :
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   年龄预测无正则化.py
@Version    :   v0.1
@Time       :   2020-05-26 17:07
@License    :   (C)Copyright 2018-2020,
@Reference  :
@Desc       :
xgboost对于年龄的预测准确率是85.29，比nn的80要强一些，但这是为什么
xgboost对于年龄极限的预测是25.6
"""
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib

matplotlib.use('TkAgg')  # 'TkAgg' can show GUI in imshow()

# ================基于XGBoost原生接口的分类=============
def exp1():
    # 加载样本数据集

    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    # 「CSV」文件字段名称
    # "creative_id","click_times","ad_id","product_id","product_category","advertiser_id","industry",
    filename = './data/all_log_300k.csv'
    csv_data = pd.read_csv(filename)
    csv_data = np.array(csv_data)
    print(csv_data.shape)
    csv_data = csv_data[:1500000, :]
    X = csv_data[:, 1:8]
    y = csv_data[:, 8].reshape([X.shape[0], 1])

    for i in range(y.shape[0]):
        y[i, 0] = y[i, 0] - 1
    y = y.reshape([X.shape[0], ])

    X = preprocessing.scale(X)

    random_state = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y,
                                                        random_state = random_state)
    y_train = y_train.astype(np.float)
    y_test = y_test.astype(np.float)

    # 算法参数
    params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'num_class': 10,
            'gamma': 0.1,
            'max_depth': 8,
            'lambda': 2,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 0.1,
            'silent': 0,
            'eta': 0.1,
            'seed': 1000,
            'nthread': 4,
    }

    plst = params.items()

    dtrain = xgb.DMatrix(X_train, y_train)  # 生成数据集格式
    num_rounds = 500
    model = xgb.train(plst, dtrain, num_rounds)  # xgboost模型训练

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))


def main():
    exp1()


if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print('Exp_XGBoost.py: whole time: {:.2f} min'.format(t_all / 60.))

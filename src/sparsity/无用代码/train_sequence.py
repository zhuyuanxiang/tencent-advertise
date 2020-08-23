# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   train_sequence.py
@Version    :   v0.1
@Time       :   2020-08-22 16:48
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import gc

from keras import Input

from src.base import tools
from src.base.config import batch_size, embedding_size, epochs, time_id_max, train_data_type
from src.model.build_model import build_single_model_api, build_single_output_api
from src.sparsity.train_day_sequence import build_inception, construct_model_single_input, single_data_reshape
from src.base.tools import show_title


def construct_model_multi_output():
    Input_creative_id = Input(shape=(time_id_max, embedding_size), name='creative_id')
    x_output = build_inception(1, Input_creative_id)
    return build_single_model_api([Input_creative_id], build_single_output_api(x_output))


def train_multi_output():
    from src.data.show_data import show_result, show_parameters

    show_title("构建网络模型")
    show_parameters()
    model = construct_model_single_input()
    model.summary()

    from src.model.save_model import save_model_m0
    show_title("保存网络模型")
    save_model_m0(model)

    from src.data.load_data import load_train_val_data
    x_train_val, y_train_val = load_train_val_data()
    from src.base.config import day_feature_idx
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


if __name__ == '__main__':
    # 运行结束的提醒
    tools.beep_end()

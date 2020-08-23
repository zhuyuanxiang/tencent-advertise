# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   save_model.py
@Version    :   v0.1
@Time       :   2020-08-20 9:52
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   模型保存
@理解：
"""
import pickle

from src.base.config import save_model, model_file_path, model_file_prefix
from src.base.tools import beep_end, show_title


# ----------------------------------------------------------------------
def main():
    pass


if __name__ == '__main__':
    main()
    beep_end()


def save_model_m0(model):
    if save_model:
        show_title("保存网络模型")
        file_name = model_file_path + model_file_prefix + 'm0.h5'
        print("保存原始模型:{} →".format(file_name), end='')
        model.save(file_name)
        print("模型保存成功。")


def save_model_m1(history, model):
    if save_model:
        show_title("保存网络模型")
        file_name = model_file_path + model_file_prefix + 'm1.bin'
        print("保存第一次训练模型:{} → ".format(file_name), end='')
        model.save_weights(file_name)
        with open(model_file_path + model_file_prefix + 'm1.pkl', 'wb') as fname:
            pickle.dump(history.history, fname)
        print("模型保存成功。")


def save_model_m2(history, model):
    if save_model:
        show_title("保存网络模型")
        file_name = model_file_path + model_file_prefix + 'm2.bin'
        print("保存第二次训练模型:{} → ".format(file_name), end='')
        model.save_weights(file_name)
        with open(model_file_path + model_file_prefix + 'm2.pkl', 'wb') as fname:
            pickle.dump(history.history, fname)
        print("模型保存成功。")

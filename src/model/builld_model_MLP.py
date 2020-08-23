# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   builld_model_MLP.py
@Version    :   v0.1
@Time       :   2020-08-14 11:26
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   多层前向神经网络
@理解：
"""
# common imports
from keras.layers import Flatten, Dropout, Dense
from keras.regularizers import l2

from src.base import tools

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
from src.model.build_model import build_single_input, build_single_output
from src.base.config import embedding_size

if __name__ == '__main__':
    # 运行结束的提醒
    tools.beep_end()


def build_mlp():
    model = build_single_input()
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(embedding_size * 4, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    return build_single_output(model)

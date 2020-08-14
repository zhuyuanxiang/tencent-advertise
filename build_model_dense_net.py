# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   build_model_dense_net.py
@Version    :   v0.1
@Time       :   2020-08-14 12:05
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   密度网络
@理解：
"""
# common imports
import config
import tools
from tensorflow import keras

from build_model import build_creative_id_input, build_embedded_creative_id, build_single_output_api, build_single_model_api


def build_dense_net():
    model_input = [build_creative_id_input()]
    x0 = build_embedded_creative_id(model_input)
    x_output = x0
    model_output = build_single_output_api(x_output)
    return build_single_model_api(model_input, model_output)

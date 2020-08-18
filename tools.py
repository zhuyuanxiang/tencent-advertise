# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   tools.py
@Version    :   v0.1
@Time       :   2020-08-11 10:50
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   工具函数
@理解：
"""


# common imports
import config


def beep_end():
    # 运行结束的提醒
    import winsound
    winsound.Beep(600, 500)
    pass


def show_figures():
    import matplotlib.pyplot as plt
    # 运行结束前显示存在的图形
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass


def show_title(message=""):
    # 输出运行模块的标题
    print('-' * 5 + '>' + message + '<' + '-' * 5)
    pass


def get_w2v_file_name():
    file_name = config.model_w2v_path + config.model_file_prefix + 'w2v.kv'
    return file_name

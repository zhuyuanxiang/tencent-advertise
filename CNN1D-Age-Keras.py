# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   CNN1D-Age-Keras.py
@Version    :   v0.1
@Time       :   2020-06-04 19:26
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# common imports
import matplotlib.pyplot as plt
import numpy as np
import winsound
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datasets import make_wave, make_line
from tools import show_title

# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

N = 100  # 数据总量


# ----------------------------------------------------------------------


def 质量测试(输入, 输出, 参数):
    sum = 0
    for n in range(N):
        sum = sum + abs(输入[n].T.dot(参数) - 输出[n])
        pass
    return sum / N


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
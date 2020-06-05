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

# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

N = 100  # 数据总量


# ----------------------------------------------------------------------
print("* 加载数据集...")

# 「CSV」文件字段名称
# "time_id","user_id_inc","user_id","creative_id_inc","creative_id","click_times","age","gender"
filename = './data/train_data.csv'
df = pd.read_csv(filename)

# y_data = (df['age'].values + df['gender'] * 10).values  # 年龄和性别一起作为目标数据
y_data = df['age'].values - 1  # 年龄作为目标数据
# y_data = df['gender'].values - 1  # 性别作为目标数据
# 选择需要的列作为输入数据
X_data = df[["time_id", "creative_id_inc", "user_id_inc", "click_times"]].values

# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
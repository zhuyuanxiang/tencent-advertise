# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   classes.py
@Version    :   v0.1
@Time       :   2020-08-23 11:53
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
from enum import Enum


class ExportDataType(Enum):
    day_fix_sequence = 0  # 每人每天访问数据为长度固定的序列
    day_statistical_sequence = 1  # 每人每天访问数据序列的统计特征
    week_fix_sequence = 2  # 每人每周访问数据为长度固定的序列
    week_statistical_sequence = 3  # 每人每周访问数据序列的统计特征
    user_fix_sequence = 4  # 每人访问数据为长度固定的序列
    user_statistical_sequence = 5  # 每人访问数据序列的统计特征



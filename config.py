# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   config.py
@Version    :   v0.1
@Time       :   2020-07-09 10:14
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
seed = 42
ad_id_max = 2264190  # 最大的广告编号=广告的种类
advertiser_id_max = 62965  # 最大的广告主编号
click_times_max = 152  # 所有素材中最大的点击次数
creative_id_max = 2481135 - 1 + 3  # 最大的素材编号 = 素材的总数量 - 1，这个编号已经修正了数据库与Python索引的区别
industry_max = 335  # 最大的产业类别编号
product_category_max = 18  # 最大的产品类别编号
product_id_max = 44313  # 最大的产品编号
time_id_max = 91
user_id_num = 900000  # 用户数
# 定制 素材库大小 = creative_id_end - creative_id_start = creative_id_num = creative_id_step_size * (1 + 3 + 1)
creative_id_step_size = 128000
creative_id_window = creative_id_step_size * 5
creative_id_begin = creative_id_step_size * 0
creative_id_end = creative_id_begin + creative_id_window

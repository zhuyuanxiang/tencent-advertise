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
# import MySQLdb
import os

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)

# ----------------------------------------------------------------------
# 定义全局通用变量
file_name = '../data/train_data_all_sparsity_v.csv'

# 与数据相关的参数
ad_id_max = 2264190  # 最大的广告编号=广告的种类
advertiser_id_max = 62965  # 最大的广告主编号(52090)
click_times_max = 152  # 所有素材中最大的点击次数
creative_id_max = 2481135 - 1 + 3  # 最大的素材编号 = 素材的总数量 - 1，这个编号已经修正了数据库与Python索引的区别
industry_max = 335  # 最大的产业类别编号(326),存在缺失值(/N)的数据
product_category_max = 18  # 最大的产品类别编号(建议对它使用众数，因为每个用户的类别数不太稀疏)
product_id_max = 44313  # 最大的产品编号(33273),存在缺失值(/N)的数据
time_id_max = 91
user_id_max = 900000  # 用户数

# 定制 素材库大小 = creative_id_end - creative_id_start = creative_id_num = creative_id_step_size * (1 + 3 + 1)
creative_id_step_size = 128000
creative_id_window = creative_id_step_size * 3
creative_id_begin = creative_id_step_size * 0
creative_id_end = creative_id_begin + creative_id_window

# 与模型相关的参数
# 参数说明：
# model_type = "Bidirectional+LSTM"  # Bidirectional+LSTM：双向 LSTM
# model_type = "LeNet"  # LeNet: 卷积神经网络
# model_type = "AlexNet"  # AlexNet: 深度卷积神经网络
# model_type = "VGG"  # VGG: 使用重复元素的神经网络
# model_type = "NiN"  # VGG: 使用重复元素的神经网络
model_type = "ResNet"  # VGG: 使用重复元素的神经网络
# model_type = "GoogLeNet"  # VGG: 使用重复元素的神经网络
# model_type = "Conv1D+LSTM"  # Conv1D+LSTM：1 维卷积神经网络 + LSTM
# model_type = "GlobalMaxPooling1D+MLP"  # GlobalMaxPooling1D+MLP：1 维全局池化层 + 多层感知机
# model_type = "LSTM"  # LSTM：循环神经网络
# model_type = "MLP"  # MLP：多层感知机
# model_type = "Conv1D"  # Conv1D：1 维卷积神经网络
# model_type = "Conv1D+MLP"  # Conv1D+MLP：1 维卷积神经网络
# model_type = "GM"  # GlobalMaxPooling1D：1 维全局池化层

learning_rate = 3e-04
epochs = 30
batch_size = 1024
max_len = 128  # {64:803109，128:882952, 256:898459, 384:899686} 个用户；{64:1983350，128:2329077} 个素材
embedding_size = 32
embedding_window = 128
show_parameter = True  # 显示模型参数

# 与训练相关的参数
field_list = [  # 输入数据处理：选择需要的列
        "user_id",
        "creative_id_inc_sparsity_hash",
        "time_id",
        "click_times",
        "product_category"
]
label_list = ['age', 'gender']
label_name = 'gender'
# conn = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='root', db='tencent', charset='utf8')
# sql = "SELECT user_id,creative_id_inc_sparsity_hash AS creative_id_inc_sparsity,time_id,click_times,age,gender " \
#       "FROM train_data_all_output ORDER BY user_id,time_id"
# df = pd.read_sql(config.sql,config.conn)
fix_period_length = 13
fix_period_days = 1
load_file_path = f'../../data/sparsity_hash/original/{creative_id_window}/'
load_file_name = load_file_path + 'train_data_all_output.csv'

data_w2v_path = f'../../data/sparsity_hash/word2vec/creative_id/{creative_id_window}/'
model_w2v_path = f'../../model/sparsity_hash/word2vec/creative_id/{creative_id_window}/'

export_data_type = 'day_sequence'
if export_data_type == 'day_sequence':
    data_file_path = '../../data/sparsity_hash/day_sequence/creative_id/{0}/{1}/'.format(
            creative_id_window, label_name)
    model_file_path = '../../model/sparsity_hash/day_sequence/word2vec/creative_id/{0}/{1}/{2}/'.format(
            creative_id_window, label_name, model_type)
elif export_data_type == 'fix_day':
    data_file_path = '../../data/sparsity_hash/fix_{0}_{1}/creative_id/{2}/{3}/'.format(
            fix_period_days, fix_period_length, creative_id_window, label_name)
    model_file_path = '../../model/sparsity_hash/fix_{0}_{1}/word2vec/creative_id/{2}/{3}/{4}/'.format(
            fix_period_days, fix_period_length, creative_id_window, label_name, model_type)
elif export_data_type == 'no_interval':
    data_file_path = '../../data/sparsity_hash/no_interval/with_repeat/creative_id/{0}/{1}/'.format(
            creative_id_window, label_name)
    model_file_path = '../../model/sparsity_hash/no_interval/with_repeat/word2vec/creative_id/{0}/{1}/{2}/'.format(
            creative_id_window, label_name, model_type)

model_file_prefix = f'embedding_{embedding_size}_{max_len}_'

# 用于拆分的基础数据
base_data_type = "基础数据集"
x_data_file_name = 'x_data'
y_data_file_name = 'y_data'

# 训练数据
train_data_type = '训练数据集'
x_train_file_name = 'x_train'
y_train_file_name = 'y_train'
train_val_data_type = '去除验证的训练数据集'
x_train_val_file_name = 'x_train_val'
y_train_val_file_name = 'y_train_val'

# train_data_type = '平衡的训练数据集'
# x_train_file_name = 'x_train_balance'
# y_train_file_name = 'y_train_balance'
# x_train_val_file_name = 'x_train_val_balance'
# y_train_val_file_name = 'y_train_val_balance'

# 验证数据
val_data_type = '验证数据集'
x_val_file_name = 'x_val'
y_val_file_name = 'y_val'
# 测试数据
test_data_type = '测试数据集'
x_test_file_name = 'x_test'
y_test_file_name = 'y_test'

# 平衡数据时，每个类别的倍数，例如：1表示1倍，即不增加数据；2表示2倍，增加1倍的数据；12表示12倍，增加11倍的数据
balance_age_list = [12, 3, 2, 3, 3, 4, 6, 13, 21, 35]
balance_gender_list = [1, 2]

# 控制开关
show_data = True  # 显示加载的数据
show_result = True  # 显示训练结果
save_model = False  # 保存训练模型

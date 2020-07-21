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
import MySQLdb

seed = 42

# ----------------------------------------------------------------------
# 定义全局通用变量
file_name = '../data/train_data_all_sparsity_v.csv'

# 与数据相关的参数
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
creative_id_window = creative_id_step_size * 3
creative_id_begin = creative_id_step_size * 0
creative_id_end = creative_id_begin + creative_id_window

# 与模型相关的参数
# 参数说明：
# model_type = "Bidirectional+LSTM"  # Bidirectional+LSTM：双向 LSTM
# model_type = "Conv1D"  # Conv1D：1 维卷积神经网络
# model_type = "Conv1D+LSTM"  # Conv1D+LSTM：1 维卷积神经网络 + LSTM
# model_type = "GlobalMaxPooling1D"  # GlobalMaxPooling1D：1 维全局池化层
# model_type = "GlobalMaxPooling1D+MLP"  # GlobalMaxPooling1D+MLP：1 维全局池化层 + 多层感知机
# model_type = "LSTM"  # LSTM：循环神经网络
# model_type = "MLP"  # MLP：多层感知机
model_type = 'Conv1D'
RMSProp_lr = 5e-04
epochs = 30
batch_size = 1024
max_len = 256  # {64:803109，128:882952, 256:898459, 384:899686} 个用户；{64:1983350，128:2329077} 个素材
embedding_size = 32
embedding_window = 128

# 与训练相关的参数
field_list = [  # 输入数据处理：选择需要的列
    "user_id",
    "creative_id_inc_sparsity_hash",
    "time_id",
    "click_times",
]
label_list = ['age', 'gender']
label_name = 'gender'
# conn = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='root', db='tencent', charset='utf8')
# sql = "SELECT user_id,creative_id_inc_sparsity_hash AS creative_id_inc_sparsity,time_id,click_times,age,gender " \
#       "FROM train_data_all_output ORDER BY user_id,time_id"
# df = pd.read_sql(config.sql,config.conn)
load_file_path = '../../save_data/sparsity_hash/original/{}/'.format(creative_id_window)
load_file_name = load_file_path + 'train_data_all_output.csv'
data_file_path = '../../save_data/sparsity_hash/no_interval/with_repeat/creative_id/{0}/{1}/'.format(creative_id_window, label_name)
data_w2v_path = '../../save_data/sparsity_hash/word2vec/no_interval/with_repeat/creative_id/{0}/'.format(creative_id_window)
model_file_path = '../../save_model/sparsity_hash/no_interval/with_repeat/word2vec/creative_id/{0}/{1}/'.format(creative_id_window, label_name)
model_w2v_path = '../../save_model/sparsity_hash/word2vec/no_interval/with_repeat/creative_id/{0}/'.format(creative_id_window)

balance_age_list = [11, 2, 1, 2, 2, 3, 5, 12, 20, 34]
balance_gender_list = [0, 1]

# -*- encoding: utf-8 -*-
"""
@Author     :
@Contact    :
@site       :
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   MLP-年龄预测-无正则化.py
@Version    :   v0.1
@Time       :   2020-05-26 17:07
@License    :   (C)Copyright 2018-2020,
@Reference  :
@Desc       :
准确率上限0.8027
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# 「CSV」文件字段名称
# "creative_id","click_times","ad_id","product_id","product_category","advertiser_id","industry",
filename = './data/all_log_300k.csv'
df = pd.read_csv(filename)
y_data = df['gender'].values - 1  # 性别作为目标数据
X_data = df[[
        "creative_id",
        "click_times",
        "ad_id",
        "product_id",
        "product_category",
        "advertiser_id",
        "industry",
]].values
X = X_data
y = y_data

# 编码目标数据，使用 sklearn 的 One-Hot 编码
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
y_one_hot = enc.fit_transform(y.reshape(-1, 1)).toarray()
# y_one_hot = enc.fit_transform(np.concatenate((y_train, y_test)).reshape(-1, 1)).toarray()
# y_train_one_hot = y_one_hot[:train_len]
# y_test_one_hot = y_one_hot[train_len:]

# csv_data = pd.read_csv(filename)
# csv_data = np.array(csv_data)
# print(csv_data.shape)
# csv_data = csv_data[:1500000, :]
# X = csv_data[:, 1:8]
# y = csv_data[:, 9].reshape([X.shape[0], 1])
# y_one_hot = np.zeros([y.shape[0], 2])
# for i in range(y.shape[0]):
#     if y[i, 0] == 1:
#         # TODO:这个One-Hot 编码是错的
#         y_one_hot[i, 0] = 0
#     if y[i, 0] == 2:
#         y_one_hot[i, 1] = 1
#     if i % 10000 == 0:
#         print(i)

y = y_one_hot
X = preprocessing.scale(X)

random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y,
                                                    random_state = random_state)
y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)

print("y_train.shape =",y_train.shape)

max_steps = 50000
learing_rate = 0.00007
batch_size = 100
log_dir = 'log_dir'


def create_layer(input_tensor, input_num, output_num, layer_name, act = tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_num, output_num], stddev = 0.1))

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.1, shape = [output_num]))

        with tf.name_scope('Wx_add_b'):
            pre_activate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', pre_activate)
        activations = act(pre_activate, name = 'activation')
        tf.summary.histogram('activations', activations)
        regularizer = tf.contrib.layers.l2_regularizer(0.001)  # 字母L2，不是数字12
        a = regularizer(weights)
        return activations, a


# 以预估值y和实际值y_data之间的均方误差,以及L2正则化作为损失


print('loss定义完成')
print('loss定义完成')
with tf.name_scope('train'):
    x = tf.placeholder(tf.float32, [None, 7], name = 'x')
    y_label = tf.placeholder(tf.float32, [None, 2], name = 'y_label')
    hidden1, h_L_1 = create_layer(x, 7, 256, 'layer1')

    hidden2, h_L_2 = create_layer(hidden1, 256, 100, 'hidden2')

    gender_age_pre, h_L_3 = create_layer(hidden2, 100, 2, 'gender_age')

    loss = tf.nn.softmax_cross_entropy_with_logits(logits = gender_age_pre, labels = y_label)
    loss = tf.reduce_mean(loss) + h_L_1 + h_L_2 + h_L_3
    train_step = tf.train.AdamOptimizer(learing_rate).minimize(loss)

k = tf.argmax(gender_age_pre, 1)
k_label = tf.argmax(y_label, 1)
pre = tf.equal(k, k_label)
acc = tf.reduce_mean(tf.cast(pre, tf.float32))

merged = tf.summary.merge_all()
saver = tf.train.Saver()
print('Saver 定义完成')

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        # print(i)  # TODO:为什么要打的 i？
        init = tf.global_variables_initializer()

        sess.run(init)  # 初始化全局
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

        for i in range(max_steps):
            if i % 50 == 0:
                saver.save(sess, log_dir + '/model.ckpt', i)
                print('{0}_step,cross_entropy={1}'.format(i, sess.run(loss, feed_dict = {
                        x: X_train, y_label: y_train
                })))
                acc1 = sess.run(acc, feed_dict = {x: X_test, y_label: y_test})
                print('test_acc:{0}'.format(acc1))
            else:
                sess.run(train_step, feed_dict = {x: X_train, y_label: y_train})

        train_writer.close()

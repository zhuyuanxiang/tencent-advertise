# _*_coding: UTF-8_*_
# 开发团队：Grophysics
# 开发人员：杨立昆的粉丝
# 开发时间：2020/5/27 22:18
# 开发工具：JetBrains
# 开发理念：Rm-Rn is reality
# _*_coding: UTF-8_*_
# 开发团队：Grophysics
# 开发人员：杨立昆的粉丝
# 开发时间：2020/5/27 21:52
# 开发工具：JetBrains
# 开发理念：Rm-Rn is reality
# _*_coding: UTF-8_*_
# 开发团队：Grophysics
# 开发人员：杨立昆的粉丝
# 开发时间：2020/5/25 10:31
# 开发工具：JetBrains
# 开发理念：Rm-Rn is reality
# _*_coding: UTF-8_*_
# 开发团队：Grophysics
# 开发人员：风清扬
# 开发时间：2019/12/25 20:26
# 文件名称：可视化框架测试.PY
# 开发工具：独孤九剑
# 开发理念：Rm-Rn is reality

# 准确率上限是0.219
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# 「CSV」文件字段名称
# "creative_id","click_times","ad_id","product_id","product_category","advertiser_id","industry",
csv_data = pd.read_csv("./年龄性别数据.csv")
csv_data = np.array(csv_data)
print(csv_data.shape)
csv_data = csv_data[:1500000, :]
X = csv_data[:, 1:8]
y = csv_data[:, 8].reshape([X.shape[0], 1])
y_onehot = np.zeros([y.shape[0], 10])
for i in range(y.shape[0]):
    a = int(y[i, 0])
    y_onehot[i, a - 1] = 1

y = y_onehot
X = preprocessing.scale(X)

random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y,
                                                    random_state = random_state)
y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)

print(y_train.shape)

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
        return activations


# 以预估值y和实际值y_data之间的均方误差,以及L2正则化作为损失


print('loss定义完成')
with tf.name_scope('train'):
    x = tf.placeholder(tf.float32, [None, 7], name = 'x')
    y_label = tf.placeholder(tf.float32, [None, 10], name = 'y_label')
    hidden1 = create_layer(x, 7, 256, 'layer1')
    gender_age_pre = create_layer(hidden1, 256, 10, 'gender_age')
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = gender_age_pre, labels = y_label)
    loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(learing_rate).minimize(loss)

k = tf.argmax(gender_age_pre, 1)
k_label = tf.argmax(y_label, 1)
pre = tf.equal(k, k_label)
acc = tf.reduce_mean(tf.cast(pre, tf.float32))

merged = tf.summary.merge_all()
saver = tf.train.Saver()
print('a')

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        print(i)
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

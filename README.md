# tencent-advertise

2020 腾迅广告算法项目流程

1.  整理数据
    1.  建立原始表结构，导入三张原始表
        1.  ad.csv
        2.  user.csv
        3.  click_log.csv
    2.  生成 sparsity, tf, idf 的相关统计信息
        1.  tf: 每个素材访问的频度，即每个素材出现的次数
        2.  idf: 每个素材
    3.  生成 user_id_inc, creative_id_inc
2.  提取特征
    1.  Word2Vec: `640000*32`, `1024000*32`
    2.  全局特征：max ( ) , min ( ) , average ( ) , mode ( ) , pca ( )
    3.  序列特征：rnn ( )
    4.  局部特征：cnn ( ), 3*32, 然后摊平
    5.  统计特征：std ( ) , skew ( ) , kurt ( ) , nuique ( ) , count ( ) , times ()
3.  搭建模型
    1.  独立模型，用于对比不同模型、不同参数、不同特征的效果
    2.  混合模型，用于提升预测的精度
4.  调整参数
5.  保存结果

注：

1.  最大值 ( max ) , 最小值 ( min ) , 平均值 ( average ) , 众数 ( mode ) , 主成分 ( pca )
2.  循环神经网络 ( rnn )
3.  卷积神经网络 ( cnn )
4.  标准差 ( std ) , 偏度 ( skew ) , 峭度 ( kurt ) , n 个唯一值 ( nunique, 即每个用户访问的素材的种类 ) , n 个访问素材 ( count, 每个用户访问素材的次数 ) , n 个素材点击次数 ( times, 每个用户访问素材的点击次数 )

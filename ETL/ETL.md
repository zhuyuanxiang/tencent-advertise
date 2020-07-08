# 数据整理过程

## C1. 导入原始数据

-   导入 `click_log.csv` 到 `click_log` 表中
    -   数据量：30082771
        -   单词个数(`creative_id`)：2481135
        -   文章个数(`user_id`)：900000
    -   将 `time` 字段改成 `time_id`
    -   扩展 `sparsity` 字段用于记录数据的稀疏性
-   导入 `ad.csv` 到 `ad_list` 表中
    -   数据量：2481135
    -   导入数据之前，需要将 `ad.csv` 中的`\N` 转换成 0
        -   `product_category` 和 `industry` 字段中存在未知数据
    -   扩展字段
        -   `creative_id_inc_sparsity`: 基于 `sparsity_value` 重新生成的 `creative_id`
        -   `creative_id_inc_tf_idf`: 基于 `tf_idf_value` 重新生成的 `creative_id`
        -   `sum_creative_id_times`: 每个素材出现的次数 = tf_value
        -   `sum_user_id_times`: 每个素材的访问用户数 = idf_value
        -   `sparsity`: 用户访问的素材种类越少，这个素材的稀疏度就越高，user_list.sum_creative_id_category
        -   `tf_value`: 素材出现的次数越多越重要，ad_list.sum_creative_id_times
        -   `idf_value`: 被访问的素材的用户数越少越重要，ad_list.sum_user_id_times
        -   `tf_idf_value`: LOG(tf_value + 1) * (LOG( 文章个数 / idf_value))
        -   `sparsity_value`: 1 * tf_idf_value / sparsity
-   导入 `user.csv` 到 `user_list` 表中
    -   数据量：900000
    -   扩展字段
        -   `sum_user_click_times` : 每个用户点击素材的次数
        -   `sum_creative_id_times` : 每个用户访问素材的次数
            -   { 访问次数：用户数} : { 512 : 899882; 256 : 898480; 128 : 883325}
        -   `sum_creative_id_category` : 每个用户访问素材的种类  = sparsity，种类越少，素材在这个用户这里的稀疏度就越高，越需要保留这个素材，才能有效分离这个用户

注：

1.  表中可能某些字段名称与关键字冲突，或者包含特殊字符，记得修改，方便后序操作
2.  导入数据的表需要主键，防止导入的数据中存在错误

## C2. 创建有效数据表

有效数据表：即`product_id` 和 `industry` 中没有 0 的数据

-   创建 `ad_list` 的有效数据表`ad_valid`
    -   数据量：1474930
    -   从 `ad_list` 中导入数据
-   创建 1600 万 的有效数据表 `all_log_valid`
    -   数据量：16411005
    -   单词个数(`creative_id`)：1474930
    -   文章个数(`user_id`)：886733
    -   从`click_log`、`ad_valid`、`user_list` 中导入数据
-   创建基于内存表的`product_id`的有效数据表，用于保存临时统计的 `product_id`数据
-   创建 700 万 的有效数据表 `all_log_valid_7m`
    -   数据量：7152231
    -   单词个数(`creative_id`)：835716
    -   文章个数(`user_id`)：807834
    -   从`all_log_valid`中导入数据
-   创建 300 万 的有效数据表 `all_log_valid_3m`
    -   数据量：3068413
    -   单词个数(`creative_id`)：449699
    -   文章个数(`user_id`)：610031
    -   从`all_log_valid_7m`中导入数据
-   创建 100 万 的有效数据表 `all_log_valid_1m`
    -   数据量：1007772
    -   单词个数(`creative_id`)：203603
        -   每个素材出现的次数超过「9个：17914、8个：20089、7个：22820、6个：26357、5个：31159、4个：37997、3个：48318、2个：65889、1个：101567」
    -   文章个数(`user_id`)：373489
        -   91天内读取素材数目超过「9个：13800、8个：17230、7个：21688、6个：28012、5个：37076、4个：51101、3个：73372、2个：112722、1个：190171」
    -   从`all_log_valid_3m`中导入数据

注1：创建的数据表尽量不使用主键，因为存储的时候需要条件检查，消耗时间；

注2：为也加快检索，可以在插入数据以后，建立索引，方便数据检索

## C03. 创建统计数据表

-   创建`all_log_valid_1m`相关的统计数据表
    -   创建 `count_number_1m_creative_id`视图：表示`all_log_valid_1m`中不同`creative_id`的数目
    -   创建`value_1m_creative_id`表：表示`all_log_valid_1m`中每个`creative_id`出现的次数，以及每个`creative_id`相比总的`creative_id`所占的比例

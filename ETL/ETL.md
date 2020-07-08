# 数据整理过程

## C.1. 导入原始数据 ( `create_original-table.sql` )

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
        -   `sum_creative_id_times`: 每个素材出现的次数 = `tf_value`
        -   `sum_user_id_times`: 每个素材的访问用户数 = `idf_value`
        -   `sparsity`: 用户访问的素材种类越少，这个素材的稀疏度就越高，`user_list`.`sum_creative_id_category`
        -   `tf_value`: 素材出现的次数越多越重要，`ad_list`.`sum_creative_id_times`
        -   `idf_value`: 被访问的素材的用户数越少越重要，`ad_list`.`sum_user_id_times`
        -   `tf_idf_value`: `LOG( tf_value + 1) * (LOG( 文章个数 / idf_value ))`
        -   `sparsity_value`: 1 * `tf_idf_value / sparsity`
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

## C.2. 创建统计数据表 ( `create-statistical-table.sql` )

### C.2.1. 生成 `tf_value`, `idf_value`, `tf_idf_value`

-   更新 `ad_list` 中的统计信息
    -   `sum_creative_id_times` →  `tf_value`
    -   `sum_user_id_times` → `idf_value`
    -   `tf_idf_value = LOG(900000 / idf_value) * (tf_value + 1) / 2481135`
        -   `idf_value + 1` : 防止出现 0 作被除数，这个数据集中不存在这个问题

### C.2.2. 生成 `sparsity`, `sparsity_value`

-   更新 `user_list` 中的统计信息
    -   `sum_creative_id_times`
    -   `sum_creative_id_category`
-   根据 `user_list` 更新 `click_log_all` 中 `user_id` 对应的 `creative_id` 的 `sparsity`
    -   `click_log_all.sparsity = user_list.sum_creative_id_category`
-   根据 `click_log_all` 中的 `creative_id` 最小值更新 `ad_list` 中  `creative_id` 对应的  `sparsity`
    -   `ad_list.sparsity = MIN( click_log_all.sparsity )`
-   更新 `ad_list` 中的统计信息
    -   `sparsity_value = LOG(1 + tf_value) * LOG(900000 / idf_value) / A.sparsity`
        -   `tf_value + 1` : 防止 LOG() 计算得到 0

## C.3. 生成辅助数据表 ( `create-sequence-table.sql` )

-   `train_creative_id_sparsity`
    -   基于 `train_creative_id_sparsity `更新 `ad_list.creative_id_inc_sparsity`
-   `train_creative_id_tf_idf`
    -   基于 `train_creative_id_tf_idf `更新 `ad_list.creative_id_inc_tf_idf`

## C.4. 创建导出数据表 ( `create-output-table.sql` )

-   `train_data_all_output`
    -   从 `click_log_all` 中导入原始数据
        -   `time_id`
        -   `user_id`
        -   `creative_id`
        -   `click_times`
    -   从  `user_list` 中更新字段
        -   `age`
        -   `gender`
    -   从 `ad_list` 中更新字段
        -   `creative_id_inc_sparsity`
        -   `creative_id_inc_tf_idf`
        -   `ad_id`
        -   `product_id`
        -   `product_category`
        -   `advertiser_id`
        -   `industry`

## C.5. 创建用于导出数据的视图 ( `create-view.sql` )

-   `train_data_all_sparsity_v` : 导出基于  `sparsity` 排序和创建的 `creative_id_inc` 的 csv 文件
-   `train_data_all_tf_idf_v` : 导出基于  `tf_idf` 排序和创建的 `creative_id_inc` 的 csv 文件


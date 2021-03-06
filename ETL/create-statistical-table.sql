/* C2. 生成数据的统计信息 */
/* C2.1. 生成 tf_value, idf_value, tf_idf_value */
/* 
 统计 click_log_all 中的 creative_id 素材
 sum_creative_id_times : 每个素材出现的次数 = tf_value
 sum_user_id_times ： 每个素材的访问用户数 = idf_value
 */
UPDATE
    ad_list AS A,
    (
        SELECT
            creative_id,
            COUNT(1) AS sum_creative_id_times,
            COUNT(DISTINCT A.user_id) AS sum_user_id_times
        FROM
            click_log_all AS A
        GROUP BY
            creative_id
    ) AS B
SET
    A.sum_creative_id_times = B.sum_creative_id_times,
    A.sum_user_id_times = B.sum_user_id_times
WHERE
    A.creative_id = B.creative_id;

/* 
 分别计算 `click_log_all` 中每个 `creative_id` 的 tf, idf, tf-idf 的值
 tf : 素材出现的次数越多越重要，单个素材出现次数/所有素材出现次数
 idf: 被访问的素材的用户数越少越重要，所有用户的数目/访问单个素材的用户数目
 sparsity: 用户访问的素材越单一，这个素材的稀疏度就越高
 */
/* 
 更新 tf 值 : (A.sum_creative_id_times+1 / 2481135) AS tf_value  不使用除法，直接使用 最大值 排名
 更新 idf 值 idf_value = (LOG(900000 / A.sum_user_id_times+1))  不计算取对数，所有计算留到 tf-idf 中完成，直接使用 最小值 排名
 注： 增加 1 是为了避免出现 0
 */
UPDATE
    ad_list
SET
    tf_value = sum_creative_id_times,
    idf_value = sum_user_id_times;

/* 
 计算 tf-idf 的值
 tf_value + 1 : 防止 LOG() 计算得到 0
 idf_value + 1 : 防止出现 0 作被除数，这个数据集中不存在这个问题
 900000 : 用户数（即文章的数目）
 */
UPDATE
    ad_list
SET
    tf_idf_value = LOG(900000 / idf_value) * (tf_value + 1) / 2481135;

/* C2.2. 生成 sparsity, sparsity_value */
/* 
 统计 click_log_all 中的 user_id 的统计数据
 sum_creative_id_times : 每个用户访问素材的次数
 sum_creative_id_category : 每个用户访问素材的种类 = sparsity
 */
UPDATE
    user_list AS A,
    (
        SELECT
            user_id,
            COUNT(1) AS sum_creative_id_times,
            COUNT(DISTINCT A.creative_id) AS sum_creative_id_category
        FROM
            click_log_all AS A
        GROUP BY
            user_id
    ) AS B
SET
    A.sum_creative_id_times = B.sum_creative_id_times,
    A.sum_creative_id_category = B.sum_creative_id_category
WHERE
    A.user_id = B.user_id;

/*
 稀疏性(Sparsity)：由每个 user_id 中的 creative_id 的种类(sum_creative_id_category)决定
 目的：影响词典的 creative_id 排序，使稀疏性强的素材排在前面，保证每个用户都有惟一的素材进行标识
 1. 通过 click_log 基于每个 user_id 更新对应的 每个 creative_id 的稀疏性
 2. 通过 tf-idf 更新每个 creative_id 的稀疏性(使用 click_log 中的 creative_id 中 sparsity 最小的值)
 3. 基于 tf-idf 的 sparsity, idf, tf 生成 train_creative_id 的 creative_id_inc
 sparsity = user_list.sum_creative_id_category, 每个用户访问素材的种类
 tf_value = ad_list.sum_creative_id_times, 每个素材出现的次数
 idf_value= ad_list.sum_user_id_times, 每个素材的访问用户数
 */
UPDATE
    click_log_all AS A,
    user_list AS B
SET
    A.sparsity = B.sum_creative_id_category
WHERE
    A.user_id = B.user_id;

/* 
 更新 sparsity 值 
 使用 click_log_all 中的 creative_id 中 sparsity 最小的值更新 ad_list 中的 creative_id 的 sparsity
 值越小越重要
 */
UPDATE
    ad_list AS A,
    (
        SELECT
            MIN(sparsity) AS sparsity,
            creative_id
        FROM
            click_log_all
        GROUP BY
            creative_id
    ) AS B
SET
    A.sparsity = B.sparsity
WHERE
    A.creative_id = B.creative_id;

/*
 更新 sparsity_value 值
 使用 sparsity, tf_idf_value 更新 sparsity_value, 值越大越重要
 1/sparsity : 度量稀疏的重要性，稀疏度为 1 的素材的重要性是 稀疏度为 2 的素材的重要性的 2 倍
 */
UPDATE
    ad_list AS A
SET
    A.sparsity_value = LOG(1 + tf_value) * LOG(900000 / idf_value) / A.sparsity;

/* 更新 sum_creative_id_classes 的值 */
UPDATE
    ad_list AS A,
    (
        SELECT
            product_id,
            product_category,
            advertiser_id,
            industry,
            count(1) AS sum_creative_id_classes
        FROM
            `ad_list`
        GROUP BY
            product_id,
            product_category,
            advertiser_id,
            industry
    ) AS B
SET
    A.sum_creative_id_classes = B.sum_creative_id_classes
WHERE
    A.product_id = B.product_id
    AND A.product_category = B.product_category
    AND A.advertiser_id = B.advertiser_id
    AND A.industry = B.industry;

/* 更新 creative_id_inc_sparsity_hash 的值 */
UPDATE
    ad_list AS A,
    (
        SELECT
            product_id,
            product_category,
            advertiser_id,
            industry,
            MIN(creative_id_inc_sparsity) AS creative_id_inc_sparsity_hash
        FROM
            `ad_list`
        GROUP BY
            product_id,
            product_category,
            advertiser_id,
            industry
    ) AS B
SET
    A.creative_id_inc_sparsity_hash = B.creative_id_inc_sparsity_hash
WHERE
    A.product_id = B.product_id
    AND A.product_category = B.product_category
    AND A.advertiser_id = B.advertiser_id
    AND A.industry = B.industry;
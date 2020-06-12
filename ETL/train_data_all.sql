/* 
 creative_id_number : 2481135
 user_id_number : 900000
 click_log : 30082771
 */
/* C1. 导入原始数据 */
/* 
 点击日志表，click_log.csv
 sparsity: 用户访问的素材越单一，这个素材的稀疏度就越高
 */
CREATE TABLE `click_log_all` (
    `time_id` int NOT NULL,
    `user_id` int NOT NULL,
    `creative_id` int NOT NULL,
    `click_times` int DEFAULT NULL,
    `sparsity` int DEFAULT NULL COMMENT '用于每个 user_id 对应的 creative_id 的稀疏性，方便后面提取 creative_id 的最小值，是个临时字段',
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`) USING BTREE
)
/*!50100 STORAGE MEMORY */
ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

/* 
 为 click_log_all 的常用查询字段单独建立索引
 因为主键只有排在最前面的字段才会被用作索引
 */
ALTER TABLE
    `tencent`.`click_log_all`
ADD
    INDEX `user_id_idx`(`user_id`) USING BTREE,
ADD
    INDEX `creative_id_idx`(`creative_id`) USING BTREE;

/* 
 素材表，ad.csv
 sum_creative_click_times：每个素材点击的次数 
 sum_creative_id_times : 每个素材出现的次数
 sum_user_id_times：每个素材的访问用户数
 sparsity: 用户访问的素材越单一，这个素材的稀疏度就越高
 tf_value : 素材出现的次数越多越重要，单个素材出现次数/所有素材出现次数
 idf_value: 被访问的素材的用户数越少越重要，所有用户的数目/访问单个素材的用户数目
 tf_idf_valu = tf * idf
 */
CREATE TABLE `ad_list` (
    `creative_id` int NOT NULL,
    `ad_id` int DEFAULT NULL,
    `product_id` int DEFAULT NULL,
    `product_category` int DEFAULT NULL,
    `advertiser_id` int DEFAULT NULL,
    `industry` int DEFAULT NULL,
    sum_creative_id_times INT DEFAULT NULL,
    sum_user_id_times INT DEFAULT NULL,
    sparsity INT DEFAULT NULL,
    tf_value INT DEFAULT NULL,
    idf_value INT DEFAULT NULL,
    tf_idf_value INT DEFAULT NULL,
    PRIMARY KEY (`creative_id`)
) ENGINE = MYISAM COMMENT = 'ad.csv' DELAY_KEY_WRITE = 1;

/* 
 用户表 user.csv
 sum_user_click_times : 每个用户点击素材的次数 
 sum_creative_id_times : 每个用户访问素材的次数
 sum_creative_id_category : 每个用户访问素材的种类，
 种类越少，素材在这个用户这里的稀疏度就越高，越需要保留这个素材，才能有效分离这个用户
 */
CREATE TABLE `user_list` (
    `user_id` int NOT NULL,
    sum_creative_id_times INT DEFAULT NULL,
    sum_creative_id_category INT DEFAULT NULL,
    `age` int NOT NULL,
    `gender` int NOT NULL,
    PRIMARY KEY (`user_id`)
) ENGINE = MYISAM COMMENT = 'user.csv' DELAY_KEY_WRITE = 1;

/* C2. 创建辅助数据表 */
/*
 创建全部字段的全部数据的临时表，用于导出所需要的数据
 */
CREATE TABLE `train_data_all_temp` (
    `time_id` int NOT NULL,
    `user_id` int NOT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `creative_id` int NOT NULL,
    `click_times` int DEFAULT NULL,
    `ad_id` int DEFAULT NULL,
    `product_id` int DEFAULT NULL,
    `product_category` int DEFAULT NULL,
    `advertiser_id` int DEFAULT NULL,
    `industry` int DEFAULT NULL,
    `age` int NOT NULL,
    `gender` int NOT NULL,
    PRIMARY KEY (`creative_id`, 'user_id', 'time_id')
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '用于导出数据的临时数据表' DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all_temp(
        time_id,
        user_id,
        creative_id,
        click_times,
        ad_id,
        product_id,
        product_category,
        advertiser_id,
        industry,
        age,
        gender
    )
SELECT
    A.time_id,
    A.user_id,
    A.creative_id,
    A.click_times,
    B.ad_id,
    B.product_id,
    B.product_category,
    B.advertiser_id,
    B.industry,
    C.age,
    C.gender
FROM
    click_log_all AS A
    INNER JOIN user_list AS C ON A.user_id = C.user_id
    INNER JOIN ad_list AS B ON A.creative_id = B.creative_id;

ALTER TABLE
    `tencent`.`train_data_all_temp`
ADD
    INDEX `user_id_idx`(`user_id`) USING BTREE,
ADD
    INDEX `creative_id_idx`(`creative_id_idx`) USING BTREE,
ADD
    INDEX `age_idx`(`age`) USING BTREE,
ADD
    INDEX `gender_idx`(`gender`) USING BTREE;

/* C3. 生成数据的统计信息 */
/* 
 统计 click_log_all 中的 creative_id 素材
 sum_creative_id_times : 每个素材出现的次数 = tf_value
 sum_user_id_times：每个素材的访问用户数 = idf_value
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
 1. 通过 click_log 更新每个 user_id 对应的 每个 creative_id 的稀疏性
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
 分别计算 `click_log_all` 中每个 `creative_id` 的 tf, idf, tf-idf 的值
 tf : 素材出现的次数越多越重要，单个素材出现次数/所有素材出现次数
 idf: 被访问的素材的用户数越少越重要，所有用户的数目/访问单个素材的用户数目
 sparsity: 用户访问的素材越单一，这个素材的稀疏度就越高
 */
/* 
 更新 tf 值 : (A.sum_creative_id_times / 30082771) AS tf_value  不使用除法，直接使用 最大值 排名
 更新 idf 值 idf_value = (LOG(900000 / A.sum_user_id_times))  不计算取对数，所有计算留到 tf-idf 中完成，直接使用 最小值 排名
 */
UPDATE
    ad_list
SET
    tf_value = sum_creative_id_times,
    idf_value = sum_user_id_times;

/* 计算 tf-idf 的值：未来可能不使用*/
UPDATE
    ad_list
SET
    tf_idf_value = (tf_value / 30082771) * (LOG(900000 / idf_value));

/* 
 更新 sparsity 值 
 使用 click_log_all 中的 creative_id 中 sparsity 最小的值
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
 创建素材词典，重新编码 creative_id 
 AVG(tf_idf_value)=0.000003395131726109348
 tf_idf_value<0.000003395131726109348 
 → creative_id_inc:351850 
 → sum_creative_id_times:8; sum_user_id_times:3
 */
DROP TABLE train_creative_id_all;

CREATE TABLE train_creative_id_all (
    creative_id_inc INT NOT NULL AUTO_INCREMENT,
    creative_id INT NOT NULL,
    sparsity INT NOT NULL,
    tf_value INT NOT NULL,
    idf_value INT NOT NULL,
    PRIMARY KEY (creative_id_inc)
);

INSERT INTO
    train_creative_id_all (creative_id, sparsity, tf_value, idf_value)
SELECT
    creative_id,
    sparsity,
    tf_value,
    idf_value
FROM
    ad_list AS A
ORDER BY
    sparsity ASC,
    tf_value DESC,
    idf_value ASC;

ALTER TABLE
    `tencent`.`train_creative_id_all`
ADD
    INDEX `creative_id_idx`(`creative_id`) USING BTREE;

/* 
 依据词典要求，更新 creative_id_inc  
 更新 train_data_all_temp 的时间有点长，但是更新好后，就可以反复使用
 更新 ad_list 的时间比较短，但是更新后，还需要联合其他表才能使用
 */
UPDATE
    train_data_all_temp AS A,
    train_creative_id_all AS B
SET
    A.creative_id_inc = B.creative_id_inc
WHERE
    A.creative_id = B.creative_id;

UPDATE
    ad_list AS A,
    train_creative_id_all AS B
SET
    A.creative_id_inc = B.creative_id_inc
WHERE
    A.creative_id = B.creative_id;

/* 创建有时间序列的最终训练数据表，数据量：30082771 */
DROP TABLE train_data_all_sequence;

CREATE TABLE train_data_all_sequence (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `time_id` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL
)
/*!50100 STORAGE MEMORY */
ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all_sequence(
        user_id_inc,
        creative_id_inc,
        time_id,
        age,
        gender
    )
SELECT
    user_id AS user_id_inc,
    creative_id_inc,
    time_id,
    age,
    gender
FROM
    train_data_all_temp
ORDER BY
    user_id,
    time_id,
    creative_id_inc;

ALTER TABLE `tencent`.`train_data_all_sequence` 
ADD INDEX `user_id_inc_idx`(`user_id_inc`) USING BTREE,
ADD INDEX `creative_id_inc_idx`(`creative_id_inc`) USING BTREE;

/* 创建无时间序列的最终训练数据表，数据量：27608868 */
DROP TABLE train_data_all_no_sequence;

CREATE TABLE `train_data_all_no_sequence` (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL
)
/*!50100 STORAGE MEMORY */
ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all_no_sequence(user_id_inc, creative_id_inc, age, gender)
SELECT
    user_id AS user_id_inc,
    creative_id_inc,
    age,
    gender
FROM
    train_data_all_temp
GROUP BY
    user_id,
    creative_id_inc
ORDER BY
    user_id,
    creative_id_inc;

ALTER TABLE `tencent`.`train_data_all_no_sequence` 
ADD INDEX `user_id_inc_idx`(`user_id_inc`) USING BTREE,
ADD INDEX `creative_id_inc_idx`(`creative_id_inc`) USING BTREE;
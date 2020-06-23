/* 
 creative_id_number : 2481135
 user_id_number : 900000
 click_log : 30082771
 */
/* 
 统计 click_log_all 中的 creative_id 素材
 sum_creative_click_times：每个素材点击的次数 
 sum_creative_id_times : 每个素材出现的次数
 sum_user_id_times：每个素材的访问用户数
 */
DROP TABLE number_creative_id_all;

CREATE TABLE number_creative_id_all (
    creative_id INT NOT NULL,
    sum_creative_id_times INT NOT NULL,
    sum_user_id_times INT NOT NULL,
    sparsity INT NULL,
    tf_value INT NULL,
    idf_value INT NULL,
    tf_idf_value INT NULL,
    PRIMARY KEY (creative_id)
);

INSERT INTO
    number_creative_id_all
SELECT
    creative_id,
    COUNT(1) AS sum_creative_id_times,
    COUNT(DISTINCT A.user_id) AS sum_user_id_times
FROM
    click_log_all AS A
GROUP BY
    creative_id;

/* 
 统计 click_log_all 中的 user_id 的统计数据
 sum_user_click_times : 每个用户点击素材的次数 
 sum_creative_id_times : 每个用户访问素材的次数
 sum_creative_id_category : 每个用户访问素材的种类
 */
DROP TABLE number_user_id_all;

CREATE TABLE number_user_id_all (
    user_id INT NOT NULL,
    sum_creative_id_times INT NOT NULL,
    sum_creative_id_category INT NOT NULL,
    PRIMARY KEY (user_id)
);

INSERT INTO
    number_user_id_all
SELECT
    user_id,
    COUNT(1) AS sum_creative_id_times,
    COUNT(DISTINCT A.creative_id) AS sum_creative_id_category
FROM
    click_log_all AS A
GROUP BY
    user_id;

/*
 稀疏性(Sparsity)：由每个 user_id 中的 creative_id 的种类(sum_creative_id_category)决定
 目的：影响词典的 creative_id 排序，使稀疏性强的素材排在前面，保证每个用户都有惟一的素材进行标识
 1. 通过 click_log 更新每个 user_id 对应的 每个 creative_id 的稀疏性
 2. 通过 tf-idf 更新每个 creative_id 的稀疏性(使用 click_log 中的 creative_id 中 sparsity 最小的值)
 3. 基于 tf-idf 的 sparsity, idf, tf 生成 train_creative_id 的 creative_id_inc
 */
UPDATE
    click_log_all AS A,
    number_user_id_all AS B
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
 插入 tf 值 : (A.sum_creative_id_times / 30082771) AS tf_value 
 不使用除法，直接使用 最大值 排名
 */
/* 
 更新 idf 值 idf_value = (LOG(900000 / A.sum_user_id_times)) 
 不计算取对数，所有计算留到 tf-idf 中完成，直接使用 最小值 排名
 */
UPDATE
    number_creative_id_all
SET
    tf_value = sum_creative_id_times,
    idf_value = sum_user_id_times;

/* 计算 tf-idf 的值：未来可能不使用*/
UPDATE
    number_creative_id_all AS B
SET
    tf_idf_value = (B.tf_value / 30082771) * (LOG(900000 / B.idf_value));

/* 
 更新 sparsity 值 
 使用 click_log_all 中的 creative_id 中 sparsity 最小的值
 值越小越重要
 */
UPDATE
    number_creative_id_all AS A,
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
    tf_idf_all AS A
ORDER BY
    sparsity ASC,
    tf_value DESC,
    idf_value ASC;

/* 
 ！提取部分用户数据时才需要这样使用
 创建顺序用户词典，访问量小的数据排在前面，重新编码 user_id
 创建逆序用户词典，访问量大的数据排在前面，重新编码 user_id */
/* 使用全部用户数据，原始 user_id 已经正确编码 */
/* 
 创建训练数据，沿用 click_log_all 中的数据，重新编码 creative_id 和 user_id  
 train_data_all_temp : 保存了所有的数据临时表，用于输出最终数据表
 train_data_all : 最终用于导出数据的表，按照需要调整字段
 train_data_all_no_time : 最终用于导出数据的表，表中没有时间序列标志，按照需要调整字段
 */
CREATE TABLE `train_data_all_temp` (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `time_id` int DEFAULT NULL,
    `creative_id` int DEFAULT NULL,
    `click_times` int DEFAULT NULL,
    `ad_id` int DEFAULT NULL,
    `product_id` int DEFAULT NULL,
    `product_category` int DEFAULT NULL,
    `advertiser_id` int DEFAULT NULL,
    `industry` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL
)
/*!50100 STORAGE MEMORY */
ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

/* 插入3000万原始数据，确保每个 user_id 的原始信息更新到表中 */
INSERT INTO
    `train_data_all_temp`(
        user_id_inc,
        time_id,
        creative_id,
        click_times,
        age,
        gender
    )
SELECT
    A.user_id AS user_id_inc,
    B.time_id,
    B.creative_id,
    B.click_times,
    A.age,
    A.gender
FROM
    user_list AS A
    INNER JOIN click_log_all AS B ON A.user_id = B.user_id;

ALTER TABLE
    `tencent`.`train_data_all_temp`
ADD
    INDEX `creative_id_idx`(`creative_id`) USING BTREE;

/* 依据词典要求，更新 creative_id_inc  */
UPDATE
    train_data_all_temp AS A,
    train_creative_id_all AS B
SET
    A.creative_id_inc = B.creative_id_inc
WHERE
    A.creative_id = B.creative_id;

/* 提取素材库中的信息更新到临时表中 */
UPDATE
    train_data_all_temp AS A,
    ad_list AS B
SET
    A.ad_id = B.ad_id,
    A.product_id = B.product_id,
    A.product_category = B.product_category,
    A.advertiser_id = B.advertiser_id,
    A.industry = B.industry
WHERE
    A.creative_id = B.creative_id;

ALTER TABLE
    `tencent`.`train_data_all_temp`
ADD
    INDEX `user_id_inc_idx`(`user_id_inc`) USING BTREE,
ADD
    INDEX `creative_id_inc_idx`(`creative_id_inc`) USING BTREE,
ADD
    INDEX `time_id_idx`(`time_id`) USING BTREE;

/* 创建最终训练数据表 */
DROP TABLE train_data_all;

CREATE TABLE `train_data_all` (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `time_id` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL
)
/*!50100 STORAGE MEMORY */
ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all
SELECT
    user_id_inc,
    creative_id_inc,
    time_id,
    age,
    gender
FROM
    train_data_all_temp
ORDER BY
    user_id_inc,
    time_id,
    creative_id_inc;

/* 创建无时间序列的最终训练数据表 */
DROP TABLE train_data_all_no_time;

CREATE TABLE `train_data_all_no_time` (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL
)
/*!50100 STORAGE MEMORY */
ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all_no_time
SELECT
    user_id_inc,
    creative_id_inc,
    age,
    gender
FROM
    train_data_all_temp
GROUP BY
    user_id_inc,
    creative_id_inc
ORDER BY
    user_id_inc,
    creative_id_inc;

/* 44313	18	62965	335	152 */
SELECT
    max(product_id),
    max(product_category),
    max(advertiser_id),
    max(industry),
    max(click_times)
FROM
    `train_data_all_temp`;

/* 33273	18	52090	326	41 */
SELECT
    count(DISTINCT product_id),
    count(DISTINCT product_category),
    count(DISTINCT advertiser_id),
    count(DISTINCT industry),
    count(DISTINCT click_times)
FROM
    `train_data_all_temp`;

/* 
 age_1: 35195; age_2: 149271; age_3: 202909; age_4: 150578; age_5: 130667;
 age_6: 101720; age_7: 66711; age_8: 31967; age_9: 19474; age_10: 11508
 */
SELECT
    count(DISTINCT user_id),
    age
FROM
    `train_data_all_temp`
GROUP BY
    age
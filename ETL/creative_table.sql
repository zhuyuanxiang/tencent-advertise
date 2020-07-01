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
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1 COMMENT = 'click_log.csv';

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
    `creative_id_inc` int DEFAULT NULL,
    `creative_id` int NOT NULL,
    `ad_id` int DEFAULT NULL,
    `product_id` int DEFAULT NULL,
    `product_category` int DEFAULT NULL,
    `advertiser_id` int DEFAULT NULL,
    `industry` int DEFAULT NULL,
    `sum_creative_id_times` int DEFAULT NULL COMMENT '每个素材出现的次数',
    `sum_user_id_times` int DEFAULT NULL COMMENT '每个素材的访问用户数',
    `sparsity` int DEFAULT NULL COMMENT '用户访问的素材越单一，这个素材的稀疏度就越高',
    `tf_value` int DEFAULT NULL COMMENT '素材出现的次数越多越重要，单个素材出现次数/所有素材出现次数',
    `idf_value` int DEFAULT NULL COMMENT '被访问的素材的用户数越少越重要，LOG(所有用户的数目/访问单个素材的用户数目)',
    `tf_idf_value` int DEFAULT NULL COMMENT 'tf_idf_valu = tf * idf',
    PRIMARY KEY (`creative_id`)
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1 COMMENT = 'ad.csv';

/* 
 用户表 user.csv
 sum_user_click_times : 每个用户点击素材的次数
 sum_creative_id_times : 每个用户访问素材的次数
 sum_creative_id_category : 每个用户访问素材的种类，种类越少，素材在这个用户这里的稀疏度就越高，越需要保留这个素材，才能有效分离这个用户
 */
CREATE TABLE `user_list` (
    `user_id_inc` int DEFAULT NULL,
    `user_id` int NOT NULL,
    `sum_user_click_times` int DEFAULT NULL COMMENT '每个用户点击素材的次数',
    `sum_creative_id_times` int DEFAULT NULL COMMENT '每个用户访问素材的次数',
    `sum_creative_id_category` int DEFAULT NULL COMMENT '每个用户访问素材的种类',
    `age` int NOT NULL,
    `gender` int NOT NULL,
    PRIMARY KEY (`user_id`)
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1 COMMENT = 'user.csv';

/* C2. 创建辅助数据表 */
/*
 创建全部字段的全部数据的临时表，用于导出所需要的数据时使用
 */
CREATE TABLE `train_data_all_temp` (
    `time_id` int NOT NULL,
    `user_id_inc` int DEFAULT NULL,
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
    train_data_all_temp
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
/* C1. 导入原始数据 */
/* 点击日志表，click_log.csv */
CREATE TABLE `click_log_all` (
  `time_id` int NOT NULL,
  `user_id` int NOT NULL,
  `creative_id` int NOT NULL,
  `click_times` int DEFAULT NULL,
  `sparsity` int DEFAULT NULL COMMENT '用于每个 user_id 对应的 creative_id 的稀疏性，方便后面提取 creative_id 的最小值，是个临时字段',
  PRIMARY KEY (`time_id`,`user_id`,`creative_id`) USING BTREE
) /*!50100 STORAGE MEMORY */ ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci DELAY_KEY_WRITE=1;
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

/* 素材表，ad.csv */
CREATE TABLE `ad_list` (
    `creative_id` int NOT NULL,
    `ad_id` int NOT NULL,
    `product_id` int NOT NULL,
    `product_category` int NOT NULL,
    `advertiser_id` int NOT NULL,
    `industry` int NOT NULL,
    PRIMARY KEY (`creative_id`)
) ENGINE = MYISAM COMMENT = 'ad.csv' DELAY_KEY_WRITE = 1;

/* 用户表 user.csv*/
CREATE TABLE `user_list` (
    `user_id` int NOT NULL,
    `age` int NOT NULL,
    `gender` int NOT NULL,
    PRIMARY KEY (`user_id`)
) ENGINE = MYISAM COMMENT = 'user.csv' DELAY_KEY_WRITE = 1;

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
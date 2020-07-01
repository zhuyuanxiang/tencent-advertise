/* C4. 创建导出数据表 */
/*
 创建全部字段的全部数据的临时表，用于导出所需要的数据
 */
DROP TABLE train_data_all_output;

CREATE TABLE `train_data_all_output` (
    `time_id` int NOT NULL,
    `user_id_inc` INT DEFAULT NULL,
    `user_id` int NOT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `creative_id` int NOT NULL,
    `click_times` int NOT NULL,
    `ad_id` int DEFAULT NULL,
    `product_id` int DEFAULT NULL,
    `product_category` int DEFAULT NULL,
    `advertiser_id` int DEFAULT NULL,
    `industry` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL,
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`) USING BTREE
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '用于导出数据的临时数据表' DELAY_KEY_WRITE = 1;

/* 插入 click_log_all 的原始数据 */
INSERT INTO
    train_data_all_output(
        time_id,
        user_id,
        creative_id,
        click_times
    )
SELECT
    time_id,
    user_id,
    creative_id,
    click_times
FROM
    click_log_all;

/* 创建 train_data_all_output 的索引字段，方便后面更新其他字段的数据 */
ALTER TABLE
    `tencent`.`train_data_all_output`
ADD
    INDEX `time_id_idx`(`time_id`) USING BTREE,
ADD
    INDEX `user_id_idx`(`user_id`) USING BTREE,
ADD
    INDEX `creative_id_idx`(`creative_id`) USING BTREE,
ADD
    INDEX `age_idx`(`age`) USING BTREE,
ADD
    INDEX `gender_idx`(`gender`) USING BTREE;

/* 基于 user_list 更新 user_id_inc, age, gender */
UPDATE
    train_data_all_output AS A,
    user_list AS B
SET
    A.user_id_inc = B.user_id_inc,
    A.age = B.age,
    A.gender = B.gender
WHERE
    A.user_id = B.user_id;

/* 
 基于 ad_list 更新 
 creative_id_inc,
 ad_id,
 product_id,
 product_category,
 advertiser_id,
 industry
 */
UPDATE
    train_data_all_output AS A,
    ad_list AS B
SET
    A.creative_id_inc = B.creative_id_inc,
    A.ad_id = B.ad_id,
    A.product_id = B.product_id,
    A.product_category = B.product_category,
    A.advertiser_id = B.advertiser_id,
    A.industry = B.industry
WHERE
    A.creative_id = B.creative_id;

/* 基于 train_ad_id_all 更新 ad_id_inc */
UPDATE
    train_data_all_output AS A,
    train_ad_id_all AS B
SET
    A.ad_id_inc = B.ad_id_inc
WHERE
    A.ad_id = B.ad_id;

/* 增加 train_data_all_output 的索引字段，方便后面编程时查询相关数据 */
ALTER TABLE
    `tencent`.`train_data_all_output`
ADD
    INDEX `user_id_inc_idx`(`user_id_inc`) USING BTREE,
ADD
    INDEX `creative_id_inc_idx`(`creative_id_inc`) USING BTREE,
ADD
    INDEX `ad_id_inc_idx`(`ad_id_inc`) USING BTREE;
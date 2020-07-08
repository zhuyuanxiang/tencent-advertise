/* 
 创建有时间序列的最终训练数据视图，数据量：30082771
 注意：不要随便双击视图，因为数据量过大会导致等待时间过长
 */
CREATE VIEW train_data_all_sequence_v AS
SELECT
    user_id_inc,
    creative_id_inc,
    time_id,
    click_times,
    ad_id,
    product_id,
    product_category,
    advertiser_id,
    industry,
    age,
    gender
FROM
    train_data_all_output
ORDER BY
    user_id_inc,
    time_id,
    creative_id_inc;

/* 创建无时间序列的最终训练数据视图，数据量：27608868 */
CREATE VIEW train_data_all_no_sequence_v AS
SELECT
    user_id_inc,
    creative_id_inc,
    time_id,
    click_times,
    ad_id,
    product_id,
    product_category,
    advertiser_id,
    industry,
    age,
    gender
FROM
    train_data_all_output
GROUP BY
    user_id_inc,
    creative_id_inc
ORDER BY
    user_id_inc,
    creative_id_inc;

/* 创建最少训练字段视图 */
CREATE VIEW train_data_all_min_complete_v
SELECT
    train_data_all_output.time_id AS time_id,
    train_data_all_output.user_id_inc AS user_id_inc,
    train_data_all_output.creative_id_inc AS creative_id_inc,
    train_data_all_output.click_times AS click_times,
    train_data_all_output.age AS age,
    train_data_all_output.gender AS gender,
    train_data_all_output.ad_id_inc AS ad_id_inc
FROM
    train_data_all_output
ORDER BY
    user_id_inc,
    time_id,
    creative_id_inc
    /* 创建有时间序列的 click_times 视图，好像可以废弃 */
    CREATE VIEW train_data_all_click_times_v AS
SELECT
    train_data_all_output.user_id_inc AS user_id_inc,
    train_data_all_output.click_times AS click_times,
    train_data_all_output.time_id AS time_id,
    train_data_all_output.age AS age,
    train_data_all_output.gender AS gender
FROM
    train_data_all_output
ORDER BY
    train_data_all_output.user_id_inc ASC,
    train_data_all_output.time_id ASC
    /* 创建有时间序列的最终训练数据表，数据量：30082771 */
    DROP TABLE train_data_all_sequence;

CREATE TABLE train_data_all_sequence (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `time_id` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all_sequence(
        user_id_inc,
        creative_id_inc,
        time_id,
        age,
        gender
    )
SELECT
    user_id_inc,
    creative_id_inc,
    time_id,
    age,
    gender
FROM
    train_data_all_output
ORDER BY
    user_id_inc,
    time_id,
    creative_id_inc;

ALTER TABLE
    `tencent`.`train_data_all_sequence`
ADD
    INDEX `user_id_inc_idx`(`user_id_inc`) USING BTREE,
ADD
    INDEX `creative_id_inc_idx`(`creative_id_inc`) USING BTREE;

/* 创建无时间序列的最终训练数据表，数据量：27608868 */
DROP TABLE train_data_all_no_sequence;

CREATE TABLE `train_data_all_no_sequence` (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `age` int DEFAULT NULL,
    `gender` int DEFAULT NULL
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all_no_sequence(user_id_inc, creative_id_inc, age, gender)
SELECT
    user_id_inc,
    creative_id_inc,
    age,
    gender
FROM
    train_data_all_output
GROUP BY
    user_id_inc,
    creative_id_inc
ORDER BY
    user_id_inc,
    creative_id_inc;

ALTER TABLE
    `tencent`.`train_data_all_no_sequence`
ADD
    INDEX `user_id_inc_idx`(`user_id_inc`) USING BTREE,
ADD
    INDEX `creative_id_inc_idx`(`creative_id_inc`) USING BTREE;
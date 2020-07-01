/* C3. 创建辅助数据表 */
/* SQL 支持的聚合(aggregate) 函数有：
 avg(), count(), first(), last(), max(), min(), sum(), group by, having 
 因此更为复杂的特征提取在代码中完成 */
/* 创建用户词典，重新编码 user_id 为 user_id_inc */
DROP TABLE `train_user_id_all`;

CREATE TABLE `train_user_id_all` (
    `user_id_inc` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    sparsity INT NOT NULL,
    PRIMARY KEY (`user_id_inc`)
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    `train_user_id_all` (`user_id`, sparsity)
SELECT
    A.`user_id`,
    A.`sum_creative_id_category`
FROM
    `user_list` AS A
ORDER BY
    `sum_creative_id_category`;

ALTER TABLE
    `tencent`.`train_user_id_all`
ADD
    INDEX `sparsity_idx`(`sparsity`) USING BTREE;

/* 基于 train_user_id_all 更新 user_list 的 user_id_inc */
UPDATE
    user_list AS A,
    train_user_id_all AS B
SET
    A.user_id_inc = B.user_id_inc
WHERE
    A.user_id = B.user_id;

/*  创建素材词典，重新编码 creative_id 为 creative_id_inc */
DROP TABLE train_creative_id_all;

CREATE TABLE train_creative_id_all (
    creative_id_inc INT NOT NULL AUTO_INCREMENT,
    creative_id INT NOT NULL,
    sparsity INT NOT NULL,
    tf_value INT NOT NULL,
    idf_value INT NOT NULL,
    PRIMARY KEY (creative_id_inc)
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

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

/* 基于 train_creative_id_all 更新 creative_id_inc */
UPDATE
    ad_list AS A,
    train_creative_id_all AS B
SET
    A.creative_id_inc = B.creative_id_inc
WHERE
    A.creative_id = B.creative_id;

/* 创建广告词典，重新编码 ad_id */
DROP TABLE `train_ad_id_all`;

CREATE TABLE `train_ad_id_all` (
    `ad_id_inc` INT NOT NULL AUTO_INCREMENT,
    `ad_id` INT NOT NULL,
    sparsity INT NOT NULL,
    PRIMARY KEY (`ad_id_inc`)
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    `train_ad_id_all` (`ad_id`, sparsity)
SELECT
    A.`ad_id`,
    A.sparsity
FROM
    `ad_list` AS A
ORDER BY
    sparsity;

ALTER TABLE
    `tencent`.`train_ad_id_all`
ADD
    INDEX `ad_id_idx`(`ad_id`) USING BTREE;

/* 基于 train_user_id_all 更新 user_id_inc */
UPDATE
    user_list AS A,
    train_user_id_all AS B
SET
    A.user_id_inc = B.user_id_inc
WHERE
    A.user_id = B.user_id;

/* 创建user_id + time_id 统计信息 */
DROP TABLE `train_user_time_all`;

CREATE TABLE `train_user_time_all` (
    `user_id` INT NOT NULL,
    `time_id` INT NOT NULL,
    day_creative_id INT NOT NULL,
    day_creative_category INT NOT NULL
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    `train_user_time_all`(
        user_id,
        time_id,
        day_creative_id,
        day_creative_category
    )
SELECT
    A.`user_id`,
    A.`time_id`,
    count(A.creative_id) AS day_creative_id,
    count(DISTINCT A.creative_id) AS day_creative_category
FROM
    `click_log_all` AS A
GROUP BY
    user_id,
    time_id;

ALTER TABLE
    `tencent`.`train_user_time_all`
ADD
    INDEX `user_id_idx`(`user_id`) USING BTREE,
ADD
    INDEX `time_id_idx`(`time_id`) USING BTREE;

/* 创建user_id + week 统计信息 */
DROP TABLE `train_user_week_all`;

CREATE TABLE `train_user_week_all` (
    `user_id` INT NOT NULL,
    `week_id` INT NOT NULL,
    week_creative_id INT NOT NULL,
    week_creative_category INT NOT NULL
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_user_week_all(
        user_id,
        week_id,
        week_creative_id,
        week_creative_category
    )
SELECT
    user_id,
    (time_id DIV 7) AS week_id,
    sum(day_creative_id) AS week_creative_id,
    sum(day_creative_category) AS week_creative_category
FROM
    `train_user_time_all`
GROUP BY
    user_id,
    week_id
ORDER BY
    user_id,
    week_id;

ALTER TABLE
    `tencent`.`train_user_week_all`
ADD
    INDEX `user_id_idx`(`user_id`) USING BTREE,
ADD
    INDEX `week_id_idx`(`week_id`) USING BTREE;

/* 统计每周访问不同素材个数的用户个数 */
SELECT
    week_creative_id,
    count(DISTINCT user_id)
FROM
    `train_user_week_all`
GROUP BY
    week_creative_id
ORDER BY
    week_creative_id;
/* C1. 导入原始数据 */
/* 点击日志表，click_log.csv */
CREATE TABLE 'click_log'(
    `time_id` int NOT NULL,
    `user_id` int NOT NULL,
    `creative_id` int NOT NULL,
    `click_times` int NOT NULL,
    PRIMARY KEY (`creative_id`, 'user_id', 'time_id')
) ENGINE = MyISAM COMMENT = 'click_log.csv' DELAY_KEY_WRITE = 1;

/* 素材表，ad.csv */
CREATE TABLE `ad_list` (
    `creative_id` int NOT NULL,
    `ad_id` int NOT NULL,
    `product_id` int NOT NULL,
    `product_category` int NOT NULL,
    `advertiser_id` int NOT NULL,
    `industry` int NOT NULL,
    PRIMARY KEY (`creative_id`)
) ENGINE = MyISAM COMMENT = 'ad.csv' DELAY_KEY_WRITE = 1;

/* 用户表 user.csv*/
CREATE TABLE `user_list` (
    `user_id` int NOT NULL,
    `age` int NOT NULL,
    `gender` int NOT NULL,
    PRIMARY KEY (`user_id`)
) ENGINE = MyISAM COMMENT = 'user.csv' DELAY_KEY_WRITE = 1;

/* C2. 创建辅助数据表 */
/* 创建基于内存的表`number_id`，用于保存统计的 `xxx_id`的数据 */
DROP TABLE number_id_all;

CREATE TABLE number_id_all (
    `id_inc` int NOT NULL AUTO_INCREMENT,
    `id` int NULL,
    `id_number` int NULL,
    PRIMARY KEY (`id_inc`)
) ENGINE = MyISAM STORAGE MEMORY;

/* 注：因为是内存表，每次系统重启后，如果还要使用这张表，需要重新插入数据，大约60秒完成 */
INSERT INTO
    number_id_all(id, id_number)
SELECT
    user_id AS id,
    COUNT(1) AS id_number
FROM
    click_log
GROUP BY
    user_id
ORDER BY
    id_number;

ALTER TABLE
    number_id_all
ADD
    INDEX `id_idx`(`id`) USING BTREE ，
ADD
    INDEX `id_number_idx`(`id_number`) USING BTREE;

/*
 创建全部字段的全部数据的表
 */
CREATE TABLE all_log
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
    click_log AS A
    INNER JOIN user_list AS C ON A.user_id = C.user_id
    INNER JOIN ad_list AS B ON A.creative_id = B.creative_id;

ALTER TABLE
    `tencent`.`all_log`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

ALTER TABLE
    `tencent`.`all_log`
ADD
    INDEX `age_idx`(`age`) USING BTREE,
ADD
    INDEX `gender_idx`(`gender`) USING BTREE;

CREATE TABLE all_log_age_1
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 1;

ALTER TABLE
    `tencent`.`all_log_age_1`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_2
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 2;

ALTER TABLE
    `tencent`.`all_log_age_2`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_3
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 3;

ALTER TABLE
    `tencent`.`all_log_age_3`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_4
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 4;

ALTER TABLE
    `tencent`.`all_log_age_4`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_5
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 5;

ALTER TABLE
    `tencent`.`all_log_age_5`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_6
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 6;

ALTER TABLE
    `tencent`.`all_log_age_6`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_7
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 7;

ALTER TABLE
    `tencent`.`all_log_age_7`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_8
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 8;

ALTER TABLE
    `tencent`.`all_log_age_8`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_9
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 9;

ALTER TABLE
    `tencent`.`all_log_age_9`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_age_10
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.age = 10;

ALTER TABLE
    `tencent`.`all_log_age_10`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_gender_1
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.gender = 1;

ALTER TABLE
    `tencent`.`all_log_gender_1`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

CREATE TABLE all_log_gender_2
SELECT
    all_log.*
FROM
    all_log
WHERE
    all_log.gender = 2;

ALTER TABLE
    `tencent`.`all_log_gender_2`
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

/* 
 创建 50 万的数据表，标准是每个用户访问素材数目小于11
 用户数： 50058
 素材数： 176538
 数据条数：  483211
 注：还可以小于 21 创建 500万数据表；小于 51 创建 1500 万数据表
 */
CREATE TABLE click_log_500k
SELECT
    A.*
FROM
    click_log AS A,
    number_id_all AS B
WHERE
    B.id_number < 11
    AND A.user_id = B.id;

ALTER TABLE
    click_log_500k
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

/* 
 创建 100 万的数据表，标准是每个用户访问素材数目以 450 为均值
 用户数：
 素材数：
 数据条数： 
 */
CREATE TABLE click_log_1m
SELECT
    A.*
FROM
    click_log AS A,
    number_id_all AS B
WHERE
    B.id_number BETWEEN 200
    AND 700
    AND A.user_id = B.id;

ALTER TABLE
    click_log_1m
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

/* 
 创建 500 万的数据表，标准是每个用户访问素材数目小于 51
 用户数： 382579
 素材数： 891856
 数据条数： 5458665
 */
CREATE TABLE click_log_5m_51
SELECT
    A.*
FROM
    click_log AS A,
    number_id_all AS B
WHERE
    A.user_id = B.id
    AND B.id_number < 51
ALTER TABLE
    click_log_5m_51
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

/* 
 创建 500 万的数据表，标准是每个用户访问素材数目在 100 和 800 之间
 用户数： 
 素材数： 
 数据条数： 
 */
CREATE TABLE click_log_5m
SELECT
    A.*
FROM
    click_log AS A,
    number_id_all AS B
WHERE
    A.user_id = B.id
    AND B.id_number BETWEEN 100
    AND 800;

ALTER TABLE
    click_log_5m
ADD
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`);

/* 
 创建 1500 万的数据表
 用户数：
 素材数：
 数据条数： 
 */
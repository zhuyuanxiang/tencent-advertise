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
/* 创建非空值的素材表 */
CREATE TABLE `ad_valid` (
    `creative_id` int NOT NULL,
    `ad_id` int NOT NULL,
    `product_id` int NOT NULL,
    `product_category` int NOT NULL,
    `advertiser_id` int NOT NULL,
    `industry` int NOT NULL
) ENGINE = MyISAM COMMENT = 'product_id>0 and industry>0' DELAY_KEY_WRITE = 1;

INSERT INTO
    ad_valid
SELECT
    *
FROM
    ad_list
WHERE
    product_id > 0
    AND industry > 0;

ALTER TABLE
    `ad_valid`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE;

/* 创建非空值的日志表(包含全部字段) 数据量：16411005*/
CREATE TABLE `all_log_valid`(
    `time_id` int NOT NULL,
    `user_id` int NOT NULL,
    `creative_id` int NOT NULL,
    `click_times` int NOT NULL,
    `ad_id` int NOT NULL,
    `product_id` int NOT NULL,
    `product_category` int NOT NULL,
    `advertiser_id` int NOT NULL,
    `industry` int NOT NULL,
    `age` int NOT NULL,
    `gender` int NOT NULL
) ENGINE = MyISAM COMMENT = '所有的字段合并在一起' DELAY_KEY_WRITE = 1;

INSERT INTO
    `all_log_valid`
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
    INNER JOIN ad_valid AS B ON B.creative_id = A.creative_id
    INNER JOIN user_list AS C ON C.user_id = A.user_id;

ALTER TABLE
    `all_log_valid`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE,
ADD
    INDEX `user_id_idx` (`user_id`) USING BTREE,
ADD
    INDEX `product_id_idx` (`product_id`) USING BTREE;

/* 创建基于内存表的`product_id`的有效数据表，用于保存临时统计的 `product_id`数据 */
CREATE TABLE `product_id_count_number` (
    `product_id` int NOT NULL,
    `product_id_number` int NULL,
    PRIMARY KEY (`product_id`),
    INDEX `product_id_number_idx` (`product_id_number`) USING BTREE
) ENGINE = MEMORY;

/* 注：因为是内存表，每次系统重启后，如果还要使用这张表，需要重新插入数据，大约60秒完成 */
INSERT INTO
    product_id_count_number
SELECT
    product_id,
    COUNT(1) AS product_id_number
FROM
    all_log_valid
GROUP BY
    product_id;

/* 创建 700万 的有效数据表 数据量：7152231*/
CREATE TABLE `all_log_valid_7m`(
    `time_id` int NOT NULL,
    `user_id` int NOT NULL,
    `creative_id` int NOT NULL,
    `click_times` int NOT NULL,
    `ad_id` int NOT NULL,
    `product_id` int NOT NULL,
    `product_category` int NOT NULL,
    `advertiser_id` int NOT NULL,
    `industry` int NOT NULL,
    `age` int NOT NULL,
    `gender` int NOT NULL
) ENGINE = MyISAM COMMENT = '所有的字段合并在一起，700万条数据' DELAY_KEY_WRITE = 1;

INSERT INTO
    `all_log_valid_7m`
SELECT
    A.*
FROM
    `all_log_valid` as A
    INNER JOIN `product_id_count_number` AS B ON B.`product_id` = A.`product_id`
WHERE
    B.`product_id_number` BETWEEN 200
    AND 70000;

ALTER TABLE
    `all_log_valid_7m`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE,
ADD
    INDEX `user_id_idx` (`user_id`) USING BTREE,
ADD
    INDEX `product_id_idx` (`product_id`) USING BTREE;

/* 创 建 大 约 300 万 有 效 数 据  */
CREATE TABLE `all_log_valid_3m`(
    `time_id` int NOT NULL,
    `user_id` int NOT NULL,
    `creative_id` int NOT NULL,
    `click_times` int NOT NULL,
    `ad_id` int NOT NULL,
    `product_id` int NOT NULL,
    `product_category` int NOT NULL,
    `advertiser_id` int NOT NULL,
    `industry` int NOT NULL,
    `age` int NOT NULL,
    `gender` int NOT NULL
) ENGINE = MyISAM COMMENT = '所有的字段合并在一起，300万条数据' DELAY_KEY_WRITE = 1;

INSERT INTO
    `all_log_valid_3m`
SELECT
    A.*
FROM
    `all_log_valid_7m` as A
    INNER JOIN `product_id_count_number` AS B ON B.`product_id` = A.`product_id`
WHERE
    B.`product_id_number` BETWEEN 400
    AND 10000;

ALTER TABLE
    `all_log_valid_3m`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE,
ADD
    INDEX `user_id_idx` (`user_id`) USING BTREE,
ADD
    INDEX `product_id_idx` (`product_id`) USING BTREE;

/*创 建 大 约 100 万 有 效 数 据*/
CREATE TABLE all_log_valid_1m AS
SELECT
    A.*
FROM
    `all_log_valid_3m` as A
    INNER JOIN `product_id_count_number` AS B ON B.`product_id` = A.`product_id`
WHERE
    B.`product_id_number` BETWEEN 400
    AND 2100;

ALTER TABLE
    `all_log_valid_1m`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE,
ADD
    INDEX `user_id_idx` (`user_id`) USING BTREE,
ADD
    INDEX `product_id_idx` (`product_id`) USING BTREE;
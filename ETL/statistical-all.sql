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
    PRIMARY KEY (creative_id)
);

INSERT INTO
    number_creative_id_all
SELECT
    creative_id,
    COUNT(1) AS sum_creative_click_times,
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
 分别计算 `click_log_all` 中每个 `creative_id` 的 tf, idf, tf-idf 的值
 tf : 素材出现的次数越多越重要，单个素材出现次数/所有素材出现次数
 idf: 被访问的素材的用户数越少越重要，所有用户的数目/访问单个素材的用户数目
 */
DROP TABLE tf_idf_all;

CREATE TABLE tf_idf_all (
    creative_id INT NOT NULL,
    tf_value FLOAT NULL,
    idf_value FLOAT NULL,
    tf_idf_value FLOAT NULL,
    PRIMARY KEY (creative_id)
);

/* 插入 tf 值 */
INSERT INTO
    tf_idf_all (creative_id, tf_value)
SELECT
    A.creative_id,
    (A.sum_creative_id_times / 30082771) AS tf_value
FROM
    number_creative_id_all AS A;

/* 插入 idf 值 */
UPDATE
    tf_idf_all AS B
SET
    idf_value = (
        SELECT
            LOG(900000 / A.sum_user_id_times)
        FROM
            number_creative_id_all AS A
        WHERE
            A.creative_id = B.creative_id
    );

/* 计算 tf-idf 的值 */
UPDATE
    tf_idf_all AS B
SET
    tf_idf_value = B.tf_value * B.idf_value;

/* 增加 creative_id_idx 的索引 */
ALTER TABLE
    tf_idf_all
ADD
    INDEX creative_id_idx (creative_id) USING BTREE;

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
    tf_idf_value FLOAT NOT NULL,
    PRIMARY KEY (creative_id_inc)
);

INSERT INTO
    train_creative_id_all (creative_id, tf_idf_value)
SELECT
    creative_id,
    tf_idf_value
FROM
    tf_idf_all AS A
ORDER by
    tf_idf_value desc;

/* 创建顺序用户词典，访问量小的数据排在前面，重新编码 user_id */
DROP TABLE train_user_id_all_asc;

CREATE TABLE train_user_id_all_asc (
    user_id_inc INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    sum_creative_id_category INT NOT NULL,
    PRIMARY KEY (user_id_inc)
);

INSERT INTO
    train_user_id_all_asc (user_id, sum_creative_id_category)
SELECT
    A.user_id,
    sum_creative_id_category
FROM
    number_user_id_all AS A
ORDER BY
    sum_creative_id_category ASC;

/* 创建逆序用户词典，访问量大的数据排在前面，重新编码 user_id */
DROP TABLE train_user_id_all_desc;

CREATE TABLE train_user_id_all_desc (
    user_id_inc INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    sum_creative_id_category INT NOT NULL,
    PRIMARY KEY (user_id_inc)
);

INSERT INTO
    train_user_id_all_desc (user_id, sum_creative_id_category)
SELECT
    A.user_id,
    sum_creative_id_category
FROM
    number_user_id_all AS A
ORDER BY
    sum_creative_id_category DESC;

/* 
 创建训练数据，沿用 click_log_all 中的数据，重新编码 creative_id 和 user_id  
 train_data_all_bak : 保存了所有的数据，但是没有按照 creative_id_inc 排序
 train_data_all : 最终用于导出数据的表，按照需要调整内容
 */
DROP TABLE train_data_all_bak;

CREATE TABLE `train_data_all_bak` (
    `user_id_inc` int DEFAULT NULL,
    `creative_id_inc` int DEFAULT NULL,
    `time_id` int DEFAULT NULL,
    `user_id` int DEFAULT NULL,
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
ENGINE = MyISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    `train_data_all_bak`(
        user_id_inc,
        user_id,
        time_id,
        creative_id,
        click_times
    )
SELECT
    A.user_id_inc,
    A.user_id,
    B.time_id,
    B.creative_id,
    B.click_times
FROM
    train_user_id_all_desc AS A
    INNER JOIN click_log_all AS B ON A.user_id = B.user_id;

ALTER TABLE
    `tencent`.`train_data_all_bak`
ADD
    INDEX `creative_id_idx`(`creative_id`) USING BTREE;

UPDATE
    train_data_all_bak AS A,
    train_creative_id_all AS B
SET
    A.creative_id_inc = B.creative_id_inc
WHERE
    A.creative_id = B.creative_id;

UPDATE
    train_data_all_bak AS A,
    user_list AS B
SET
    A.age = B.age,
    A.gender = B.gender
WHERE
    A.user_id = B.user_id;

UPDATE
    train_data_all_bak AS A,
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
    `tencent`.`train_data_all_bak`
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
ENGINE = MyISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all
SELECT
    user_id_inc,
    creative_id_inc,
    time_id,
    age,
    gender
FROM
    train_data_all_bak
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
ENGINE = MyISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1;

INSERT INTO
    train_data_all_no_time
SELECT
    user_id_inc,
    creative_id_inc,
    age,
    gender
FROM
    train_data_all_bak
ORDER BY
    user_id_inc,
    creative_id_inc;
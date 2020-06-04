/* 表示`all_log_valid_1m`中不同`creative_id`的数目 */
/* 表示`all_log_valid_1m`中不同`user_id`的数目 */
CREATE VIEW `count_number_1m` AS
SELECT
    COUNT(DISTINCT A.`creative_id`) AS `creative_id_count_number`,
    Count(DISTINCT A.`user_id`) AS `user_id_count_number`
FROM
    all_log_valid_1m AS A;

/* 表示`all_log_valid_1m`中每个`creative_id`素材出现的次数 */
DROP TABLE `number_creative_id_1m`;

CREATE TABLE `number_creative_id_1m` (
    `creative_id` INT NOT NULL,
    `creative_id_number` INT NOT NULL,
    PRIMARY KEY (`creative_id`)
);

INSERT INTO
    `number_creative_id_1m` (`creative_id`, `creative_id_number`)
SELECT
    `creative_id`,
    count(1) AS `creative_id_number`
FROM
    `all_log_valid_1m` AS A
GROUP BY
    `creative_id`;

/* 表示`all_log_valid_1m`中每个`user_id`访问素材的次数 */
DROP TABLE `number_user_id_1m`;

CREATE TABLE `number_user_id_1m` (
    `user_id` INT NOT NULL,
    `user_id_number` INT NOT NULL,
    PRIMARY KEY (`user_id`)
);

INSERT INTO
    `number_user_id_1m` (`user_id`, `user_id_number`)
SELECT
    `user_id`,
    count(1) AS `user_id_number`
FROM
    `all_log_valid_1m` AS A
GROUP BY
    `user_id`;

/* 
 分别计算 `all_log_valid_1m`中每个`creative_id`的 `tf`,`idf`,`tf-idf`的值
 */
DROP TABLE `tf_idf_1m`;

CREATE TABLE `tf_idf_1m` (
    `creative_id` INT NOT NULL,
    `tf_value` FLOAT NULL,
    `idf_value` FLOAT NULL,
    `tf_idf_value` FLOAT NULL,
    PRIMARY KEY (`creative_id_inc`)
);

/* 插入 tf 值 */
INSERT INTO
    `tf_idf_1m` (`creative_id`, `tf_value`)
SELECT
    A.`creative_id` AS `creative_id`,
    (COUNT(A.`creative_id`) / 1007772) AS `tf_value`
FROM
    `all_log_valid_1m` AS A
GROUP BY
    A.`creative_id`;

/* 插入 idf 值 */
UPDATE
    `tf_idf_1m` AS B
SET
    idf_value = (
        SELECT
            LOG(373489 / COUNT(DISTINCT user_id))
        FROM
            `all_log_valid_1m` AS A
        WHERE
            A.creative_id = B.creative_id
        GROUP BY
            A.creative_id
    );

/* 计算 tf-idf 的值 */
UPDATE
    `tf_idf_1m` AS B
SET
    tf_idf_value = B.tf_value * B.idf_value;

/* 增加 `creative_id_idx` 的索引 */
ALTER TABLE
    `tf_idf_1m`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE;

/* 创建素材词典，重新编码 creative_id */
DROP TABLE `train_creative_id`;

CREATE TABLE `train_creative_id` (
    `creative_id_inc` INT NOT NULL AUTO_INCREMENT,
    `creative_id` INT NOT NULL,
    PRIMARY KEY (`creative_id_inc`)
);

INSERT INTO
    `train_creative_id` (`creative_id`)
SELECT
    `creative_id`
FROM
    `tf_idf_1m` AS A
ORDER by
    `tf_idf_value` desc;

/* 创建用户词典，重新编码 user_id */
DROP TABLE `train_user_id`;

CREATE TABLE `train_user_id` (
    `user_id_inc` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    PRIMARY KEY (`user_id_inc`)
);

INSERT INTO
    `train_user_id` (`user_id`)
SELECT
    A.`user_id`
FROM
    `number_user_id_1m` AS A
ORDER BY
    user_id_number DESC;

/* 创建训练数据，沿用 `all_log_valid_1m` 中的数据，重新编码 `creative_id` 和 `user_id`  */
DROP TABLE `train_data`;

CREATE TABLE `train_data`
SELECT
    A.time_id,
    C.user_id_inc,
    A.user_id,
    B.creative_id_inc,
    A.creative_id,
    A.click_times,
    A.age,
    A.gender
FROM
    all_log_valid_1m AS A
    INNER JOIN train_creative_id as B ON B.creative_id = A.creative_id
    INNER JOIN train_user_id AS C on C.user_id = A.user_id;

/* 创建训练 Word2Vec 的数据 */
CREATE TABLE train_data_word_vec
SELECT
    *
FROM
    (
        SELECT
            A.creative_id_inc AS creative_id_in,
            B.creative_id_inc AS creative_id_out
        FROM
            train_data_20k AS A
            INNER JOIN train_data_20k AS B ON B.creative_id <> A.creative_id
            AND B.user_id = A.user_id
    ) AS C
GROUP BY
    creative_id_in,
    creative_id_out
HAVING
    count(1) = 1
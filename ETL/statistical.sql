/* 表示`all_log_valid_1m`中不同`creative_id`的数目 */
CREATE VIEW `count_number_1m_creative_id` AS
SELECT
    COUNT(DISTINCT A.`creative_id`) AS `creative_id_count_number`
FROM
    all_log_valid_1m AS A;

/* 表示`all_log_valid_1m`中不同`user_id`的数目 */
CREATE VIEW `count_number_1m_user_id` AS
SELECT
    Count(DISTINCT A.`user_id`) AS `user_id_count_number`
FROM
    `all_log_valid_1m` AS A;

/* 表示`all_log_valid_1m`中每个`creative_id`出现的次数，
 以及每个`creative_id`相比总的`creative_id`所占的比例，
 出现频率少于10次，即出现比例小于 0.00001 的单词放弃 */
DROP TABLE `tf_idf_1m`;

CREATE TABLE `tf_idf_1m` (
    `creative_id_inc` INT NOT NULL AUTO_INCREMENT,
    `creative_id` INT NOT NULL,
    `tf_value` FLOAT NULL,
    `idf_value` FLOAT NULL,
    PRIMARY KEY (`creative_id_inc`)
);

/* 插入 tf 值 */
INSERT INTO
    `tf_idf_1m`(`creative_id`, `tf_value`)
SELECT
    A.`creative_id` AS `creative_id`,
    (COUNT(A.`creative_id`) / 1007772) AS `tf_value`
FROM
    `all_log_valid_1m` AS A
GROUP BY
    A.`creative_id`
HAVING
    (COUNT(A.`creative_id`) > 10);

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

ALTER TABLE
    `tf_idf_1m`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE;

/* 
 表示`all_log_valid_1m`中每个`user_id`访问素材的次数
 剔除91天内访问广告次数小于10次的用户
 */
DROP TABLE `user_id_inc_1m`;

CREATE TABLE `user_id_inc_1m` (
    `user_id_inc` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    `user_id_number` INT NOT NULL,
    PRIMARY KEY (`user_id_inc`)
);

INSERT INTO
    `user_id_inc_1m` (`user_id`, `user_id_number`)
SELECT
    `user_id`,
    count(1) AS `user_id_number`
FROM
    `all_log_valid_1m` AS A
    INNER JOIN `tf_idf_1m` AS B ON B.creative_id = A.creative_id
GROUP BY
    `user_id`
HAVING
    (count(1) > 10);

/* 创建训练数据 */
DROP TABLE `train_data`;

CREATE TABLE `train_data`
SELECT
    A.time_id,
    B.creative_id_inc,
    C.user_id_inc,
    A.click_times,
    A.gender
FROM
    all_log_valid_1m AS A
    INNER JOIN tf_idf_1m AS B ON B.creative_id = A.creative_id
    INNER JOIN user_id_inc_1m AS C ON A.user_id = C.user_id;
/* 
    统计 `click_log_5m_51` 中的 `creative_id` 素材
    sum_creative_click_times：每个素材出现的次数 
    sum_user_id_times：每个素材的访问用户数
 */
DROP TABLE `number_creative_id_5m_51`;

CREATE TABLE `number_creative_id_5m_51` (
    `creative_id` INT NOT NULL,
    `sum_creative_click_times` INT NOT NULL,
    `sum_user_id_times` INT NOT NULL,
    PRIMARY KEY (`creative_id`)
);

INSERT INTO
    `number_creative_id_5m_51`
SELECT
    `creative_id`,
    SUM(A.click_times) AS `sum_creative_click_times`,
    COUNT(DISTINCT A.user_id) AS `sum_user_id_times`
FROM
    `click_log_5m_51` AS A
GROUP BY
    `creative_id`;

/* 
    统计 `click_log_5m_51` 中的 `user_id` 的统计数据
    sum_user_click_times : 每个用户访问素材的次数 
    sum_creative_id_times : 每个用户访问素材的种类
 */
DROP TABLE `number_user_id_5m_51`;

CREATE TABLE `number_user_id_5m_51` (
    `user_id` INT NOT NULL,
    `sum_user_click_times` INT NOT NULL,
    `sum_creative_id_times` INT NOT NULL,
    PRIMARY KEY (`user_id`)
);

INSERT INTO
    `number_user_id_5m_51`
SELECT
    `user_id`,
    SUM(A.click_times) AS `sum_user_click_times`,
    COUNT(DISTINCT A.creative_id) AS `sum_creative_id_times`
FROM
    `click_log_5m_51` AS A
GROUP BY
    `user_id`;

/* 
    分别计算 `click_log_5m_51`中每个`creative_id`的 `tf`,`idf`,`tf-idf`的值
    tf : 素材出现的次数越多越重要，单个素材出现次数/所有素材出现次数
    idf: 访问素材的用户越少越重要，所有用户的数目/访问单个素材的用户数目
 */
DROP TABLE `tf_idf_5m_51`;

CREATE TABLE `tf_idf_5m_51` (
    `creative_id` INT NOT NULL,
    `tf_value` FLOAT NULL,
    `idf_value` FLOAT NULL,
    `tf_idf_value` FLOAT NULL,
    PRIMARY KEY (`creative_id`)
);

/* 插入 tf 值 */
INSERT INTO
    `tf_idf_5m_51` (`creative_id`, `tf_value`)
SELECT
    A.creative_id,
    (A.sum_creative_click_times / 1075963) AS tf_value
FROM
    number_creative_id_5m_51 AS A;

/* 插入 idf 值 */
UPDATE
    `tf_idf_5m_51` AS B
SET
    idf_value = (
        SELECT
            LOG(373489 / A.sum_user_id_times)
        FROM
            `number_creative_id_5m_51` AS A
        WHERE
            A.creative_id = B.creative_id
    );

/* 计算 tf-idf 的值 */
UPDATE
    `tf_idf_5m_51` AS B
SET
    tf_idf_value = B.tf_value * B.idf_value;

/* 增加 `creative_id_idx` 的索引 */
ALTER TABLE
    `tf_idf_5m_51`
ADD
    INDEX `creative_id_idx` (`creative_id`) USING BTREE;

/* 创建素材词典，重新编码 creative_id */
DROP TABLE `train_creative_id_5m_51`;

CREATE TABLE `train_creative_id_5m_51` (
    `creative_id_inc` INT NOT NULL AUTO_INCREMENT,
    `creative_id` INT NOT NULL,
    PRIMARY KEY (`creative_id_inc`)
);

INSERT INTO
    `train_creative_id_5m_51` (`creative_id`)
SELECT
    `creative_id`
FROM
    `tf_idf_5m_51` AS A
ORDER by
    `tf_idf_value` desc;

/* 创建用户词典，重新编码 user_id */
DROP TABLE `train_user_id_5m_51`;

CREATE TABLE `train_user_id_5m_51` (
    `user_id_inc` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    PRIMARY KEY (`user_id_inc`)
);

INSERT INTO
    `train_user_id_5m_51` (`user_id`)
SELECT
    A.`user_id`
FROM
    `number_user_id_5m_51` AS A
ORDER BY
    sum_user_click_times DESC;

/* 创建训练数据，沿用 `click_log_5m_51` 中的数据，重新编码 `creative_id` 和 `user_id`  */
DROP TABLE `train_data_5m_51`;

CREATE TABLE `train_data_5m_51`
SELECT
    A.time_id,
    C.user_id_inc,
    A.user_id,
    B.creative_id_inc,
    A.creative_id,
    A.click_times,
    D.age,
    D.gender
FROM
    click_log_5m_51 AS A
    INNER JOIN train_creative_id_5m_51 as B ON B.creative_id = A.creative_id
    INNER JOIN train_user_id_5m_51 AS C ON C.user_id = A.user_id
    INNER JOIN user_list AS D ON D.user_id=A.user_id
ORDER BY
    C.user_id_inc,A.time_id;
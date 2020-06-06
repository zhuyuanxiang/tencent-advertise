/* 表示`all_log_valid_1m`中不同`creative_id`的数目 */
/* 表示`all_log_valid_1m`中不同`user_id`的数目 */
CREATE VIEW `count_number_1m` AS
SELECT
    COUNT(DISTINCT A.`creative_id`) AS `creative_id_count_number`,
    Count(DISTINCT A.`user_id`) AS `user_id_count_number`
FROM
    all_log_valid_1m AS A;
/* 表示`all_log_valid_1m`中不同`creative_id`的数目 */
CREATE VIEW `count_creative_id_1m` AS
SELECT
    COUNT(1) AS `creative_id_count`
FROM
    number_creative_id_1m AS A

/* 表示`all_log_valid_1m`中不同`user_id`的数目 */
CREATE VIEW `count_user_id_1m` AS
SELECT
    COUNT(1) AS `user_id_count`
FROM
    number_user_id_1m;

/* 
    表示`all_log_valid_1m`中所有`creative_id`素材被访问的数目
    表示`all_log_valid_1m`中所有`user_id`访问素材的数目
    两者是等价的，都表示总的素材访问数
*/
CREATE VIEW `sum_creative_id_1m` AS
SELECT
    SUM(A.sum_creative_click_times) AS creative_id_sum
FROM
    number_creative_id_1m AS A
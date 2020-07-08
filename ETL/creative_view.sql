/* C.5. 创建用于导出数据的视图 */
/* 注意：不要随便双击视图，因为数据量 ( 30082771 ) 过大会导致等待时间过长 */
CREATE VIEW train_data_all_sparsity_v AS
SELECT
    user_id,
    creative_id_inc_sparsity,
    time_id,
    click_times,
    age,
    gender
FROM
    train_data_all_output
ORDER BY
    user_id,
    time_id,
    creative_id_inc_sparsity;

CREATE VIEW train_data_all_tf_idf_v AS
SELECT
    user_id,
    creative_id_inc_tf_idf,
    time_id,
    click_times,
    age,
    gender
FROM
    train_data_all_output
ORDER BY
    user_id,
    time_id,
    creative_id_inc_tf_idf;
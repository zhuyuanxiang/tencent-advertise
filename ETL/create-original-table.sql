/* 
 creative_id_number : 2481135
 user_id_number : 900000
 age:{
 1:35195:3.9%:12, 2:149271:16.6%:3, 3:202909:22.5%:2, 4:150578:16.7%:3, 5:130667:14.5%:3, 
 6:101720:11.3%:4, 7:66711:7.4%:6, 8:31967:3.6%:13, 9:19474:2.2%:21, 10:11508:1.3%:35}
 463.3
 gender:{1:602610; 2:297390}
 click_log : 30082771
 age:{
 1:1392097:4.6%:4.8, 2:5142384:17.1%:1.3, 3:6586194:21.9%:1, 4:4907754:16.3%:1.3, 5:4295201:14.3%:1.5, 
 6:3340626:11.1%,2.0, 7:2204348:7.3%:3, 8:1065498:3.5%:6.3, 9:685085:2.3%:9.5, 10:463584:1.5%:14.6}
 */
/* C1. 导入原始数据 */
/* 
 点击日志表，click_log.csv
 */
CREATE TABLE `click_log_all` (
    `time_id` int NOT NULL,
    `user_id` int NOT NULL,
    `creative_id` int NOT NULL,
    `click_times` int DEFAULT NULL,
    `sparsity` int DEFAULT NULL COMMENT '用于每个 user_id 对应的 creative_id 的稀疏性，方便后面提取 creative_id 的最小值，是个临时字段',
    PRIMARY KEY (`time_id`, `user_id`, `creative_id`) USING BTREE
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1 COMMENT = 'click_log.csv';

/* 
 为 click_log_all 的常用查询字段单独建立索引
 因为主键只有排在最前面的字段才会被用作索引
 */
ALTER TABLE
    `tencent`.`click_log_all`
ADD
    INDEX `user_id_idx`(`user_id`) USING BTREE,
ADD
    INDEX `creative_id_idx`(`creative_id`) USING BTREE;

/* 
 素材表，ad.csv
 */
CREATE TABLE `ad_list` (
    `creative_id_inc_sparsity` int DEFAULT NULL COMMENT '根据 sparsity_value 生成的 creative_id_inc',
    `creative_id_inc_tf_idf` int DEFAULT NULL COMMENT '根据 tf_idf_value 生成的 creative_id_inc',
    `creative_id` int NOT NULL,
    `ad_id` int DEFAULT NULL,
    `product_id` int DEFAULT NULL COMMENT '存在缺失值(/N)的数据',
    `product_category` int DEFAULT NULL,
    `advertiser_id` int DEFAULT NULL,
    `industry` int DEFAULT NULL COMMENT '存在缺失值(/N)的数据',
    `sum_creative_id_times` int DEFAULT NULL COMMENT '每个素材出现的次数',
    `sum_user_id_times` int DEFAULT NULL COMMENT '每个素材的访问用户数',
    `sparsity` int DEFAULT NULL COMMENT '用户访问的素材越单一，这个素材的稀疏度就越高',
    `sparsity_value` double DEFAULT NULL COMMENT 'sparsity_value=LOG(所有素材的数目/sparsity)',
    `tf_value` int DEFAULT NULL COMMENT '素材出现的次数越多越重要',
    `idf_value` int DEFAULT NULL COMMENT '被访问的素材的用户数越少越重要',
    `tf_idf_value` double DEFAULT NULL COMMENT 'tf_idf_value = tf(单个素材出现次数) * idf(LOG(所有用户的数目/访问单个素材的用户数目))',
    `sum_creative_id_classes` int DEFAULT NULL COMMENT '每个素材所在类别中的素材数目,类别基于product_id,product_category,advertiser_id,industry分类',
    PRIMARY KEY (`creative_id`)
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1 COMMENT = 'ad.csv';

/* 
 用户表 user.csv
 */
CREATE TABLE `user_list` (
    `user_id` int NOT NULL,
    `sum_user_click_times` int DEFAULT NULL COMMENT '每个用户点击素材的次数',
    `sum_creative_id_times` int DEFAULT NULL COMMENT '每个用户访问素材的次数',
    `sum_creative_id_category` int DEFAULT NULL COMMENT '每个用户访问素材的种类',
    `age` int NOT NULL,
    `gender` int NOT NULL,
    PRIMARY KEY (`user_id`)
) ENGINE = MYISAM DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci DELAY_KEY_WRITE = 1 COMMENT = 'user.csv';
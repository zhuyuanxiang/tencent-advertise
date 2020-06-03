# 数据整理过程

## C1. 导入原始数据

-   导入 `click_log.csv` 到 `click_log` 表中
    -   数据量：30082771
    -   单词个数(`creative_id`)：2481135
    -   文章个数(`user_id`)：900000
    -   将 `time` 字段改成 `time_id`
-   导入 `ad.csv` 到 `ad_list` 表中
    -   数据量：2481135
    -   导入数据之前，需要将 `ad.csv` 中的`\N` 转换成 0
-   导入 `user.csv` 到 `user_list` 表中
    -   数据量：900000

注1：表中可能某些字段名称与关键字冲突，或者包含特殊字符，记得修改，方便后序操作

注2：导入数据的表需要主键，防止导入的数据中存在错误

## C2. 创建有效数据表

有效数据表：即`product_id` 和 `industry` 中没有 0 的数据

-   创建 `ad_list` 的有效数据表`ad_valid`
    -   数据量：1474930
    -   从 `ad_list` 中导入数据
-   创建 1600 万 的有效数据表 `all_log_valid`
    -   数据量：16411005
    -   单词个数(`creative_id`)：1474930
    -   文章个数(`user_id`)：886733
    -   从`click_log`、`ad_valid`、`user_list` 中导入数据
-   创建基于内存表的`product_id`的有效数据表，用于保存临时统计的 `product_id`数据
-   创建 700 万 的有效数据表 `all_log_valid_7m`
    -   数据量：7152231
    -   单词个数(`creative_id`)：835716
    -   文章个数(`user_id`)：807834
    -   从`all_log_valid`中导入数据
-   创建 300 万 的有效数据表 `all_log_valid_3m`
    -   数据量：3068413
    -   单词个数(`creative_id`)：449699
    -   文章个数(`user_id`)：610031
    -   从`all_log_valid_7m`中导入数据
-   创建 100 万 的有效数据表 `all_log_valid_1m`
    -   数据量：1007772
    -   单词个数(`creative_id`)：203603
    -   文章个数(`user_id`)：373489
    -   从`all_log_valid_3m`中导入数据

注1：创建的数据表尽量不使用主键，因为存储的时候需要条件检查，消耗时间；

注2：为也加快检索，可以在插入数据以后，建立索引，方便数据检索

## C03. 创建统计数据表

-   创建`all_log_valid_1m`相关的统计数据表
    -   创建 `count_number_1m_creative_id`视图：表示`all_log_valid_1m`中不同`creative_id`的数目
    -   创建`value_1m_creative_id`表：表示`all_log_valid_1m`中每个`creative_id`出现的次数，以及每个`creative_id`相比总的`creative_id`所占的比例

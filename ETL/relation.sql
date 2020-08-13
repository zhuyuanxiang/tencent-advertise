/* 分析数据之间的关系*/
/*素材是惟一的，是主键*/
SELECT
    COUNT (1)
FROM
    ad
GROUP BY
    creative_id,
    ad_id
HAVING
    COUNT (1) > 1;

/* 广告与其他字段的关系 */
/*一个广告只能属于一个产品，一个产品类别*/
SELECT
    COUNT(1),
    product_category
FROM
    (
        SELECT
            COUNT(1),
            ad_id,
            product_category
        FROM
            ad
        GROUP BY
            ad_id
    ) AS A
GROUP BY
    A.ad_id
HAVING
    COUNT(1) > 1
LIMIT
    0, 10;

/*一个广告只属于一个广告业主*/
SELECT
    COUNT(1),
    advertiser_id
FROM
    (
        SELECT
            COUNT(1),
            ad_id,
            advertiser_id
        FROM
            ad
        GROUP BY
            ad_id
    ) AS A
GROUP BY
    A.ad_id
HAVING
    COUNT(1) > 1
LIMIT
    0, 10;

/*一个广告只属于一个产业*/
SELECT
    COUNT(1),
    industry
FROM
    (
        SELECT
            COUNT(1),
            ad_id,
            industry
        FROM
            ad
        GROUP BY
            ad_id
    ) AS A
GROUP BY
    A.ad_id
HAVING
    COUNT(1) > 1
LIMIT
    0, 10;

SELECT
    *
FROM
    ad
WHERE
    product_id = 59;

SELECT
    *
FROM
    ad
LIMIT
    0, 1000;

SELECT
    count(1)
FROM
    all_log_age_1;

SELECT
    count(1)
FROM
    all_log_age_2;

SELECT
    count(1)
FROM
    all_log_age_3;

SELECT
    count(1)
FROM
    all_log_age_4;

SELECT
    count(1)
FROM
    all_log_age_5;

SELECT
    count(1)
FROM
    all_log_age_6;

SELECT
    count(1)
FROM
    all_log_age_7;

SELECT
    count(1)
FROM
    all_log_age_8;

SELECT
    count(1)
FROM
    all_log_age_9;

SELECT
    count(1)
FROM
    all_log_age_10;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_1;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_2;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_3;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_4;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_5;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_6;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_7;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_8;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_9;

SELECT
    count(DISTINCT user_id)
FROM
    all_log_age_10;

SELECT
    count(DISTINCT user_id)
FROM
    `all_log_gender_1`;

SELECT
    count(DISTINCT user_id)
FROM
    `all_log_gender_2`;
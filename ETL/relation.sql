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
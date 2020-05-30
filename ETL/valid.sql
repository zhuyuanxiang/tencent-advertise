/* product_id, industry 没有丢失标签的数据*/
CREATE VIEW ad_valid AS
SELECT
  *
FROM
  ad
WHERE
  product_id <> -1
  AND industry <> -1;
SELECT
  *
FROM
  ad_valid
LIMIT
  0, 10;

/* 去除所有缺失数据 */
/* 创 建 大 约 1500 万 有 效 数 据  */
CREATE VIEW all_log_valid_15m AS
SELECT
  *
FROM
  all_log A
WHERE
  A.product_id > = 0
  AND A.industry > = 0;
  /* 创 建 大 约 700 万 有 效 数 据  */
  CREATE VIEW all_log_valid_7m AS
SELECT
  L.*
FROM
  all_log_valid_15m L,
  (
    SELECT
      A.product_id
    FROM
      all_log_valid_15m A
    GROUP BY
      A.product_id
    HAVING
      COUNT (1) BETWEEN 200
      AND 70000
  ) B
WHERE
  L.product_id = B.product_id;
  /* 创 建 大 约 300 万 有 效 数 据  */
  CREATE VIEW all_log_valid_3m AS
SELECT
  L.*
FROM
  all_log_valid_7m L,
  (
    SELECT
      A.product_id
    FROM
      all_log_valid_7m A
    GROUP BY
      A.product_id
    HAVING
      COUNT (1) BETWEEN 400
      AND 10000
  ) B
WHERE
  L.product_id = B.product_id;
  /*创 建 大 约 200 万 有 效 数 据*/
  CREATE VIEW all_log_valid_2m AS
SELECT
  L.*
FROM
  all_log_valid_3m L,
  (
    SELECT
      A.product_id
    FROM
      all_log_valid_3m A
    GROUP BY
      A.product_id
    HAVING
      COUNT (1) BETWEEN 400
      AND 5000
  ) B
WHERE
  L.product_id = B.product_id;
  /*创 建 大 约 100 万 有 效 数 据*/
  CREATE VIEW all_log_valid_1m AS
SELECT
  L.*
FROM
  all_log_valid_2m L,
  (
    SELECT
      A.product_id
    FROM
      all_log_valid_2m A
    GROUP BY
      A.product_id
    HAVING
      COUNT (1) BETWEEN 400
      AND 2100
  ) B
WHERE
  L.product_id = B.product_id;
  /*创 建 大 约 70 万 有 效 数 据*/
  CREATE VIEW all_log_valid_700k AS
SELECT
  L.*
FROM
  all_log_valid_1m L,
  (
    SELECT
      A.product_id
    FROM
      all_log_valid_1m A
    GROUP BY
      A.product_id
    HAVING
      COUNT (1) BETWEEN 400
      AND 1500
  ) B
WHERE
  L.product_id = B.product_id;
  /*创 建 大 约 30 万 有 效 数 据*/
  CREATE VIEW all_log_valid_300k AS
SELECT
  L.*
FROM
  all_log_valid_3m L,
  (
    SELECT
      A.product_id
    FROM
      all_log_valid_700k A
    GROUP BY
      A.product_id
    HAVING
      COUNT (1) BETWEEN 400
      AND 800
  ) B
WHERE
  L.product_id = B.product_id;

/* 将 「 gender 」 和 「 age 」 生 成 1 - of - K 编 码 */
SELECT
  A.gender,
  A.age,
  SUBSTR(
    CONCAT(
      REPEAT ('0', 20),
      '1',
      REPEAT ('0', 20)
    ),
    21 - (A.gender - 1) * 10 - (A.age - 1),
    20
  )
FROM
  user_id A;
  /* 将 `user_id` 表 中 新 增 的 `code` 字 段 基 于 「 gender 」 和 「 age 」 生 成 1 - of - K 编 码 */
UPDATE
  user_id A
SET
  code_id = SUBSTR (
    CONCAT (
      REPEAT ('0', 20),
      '1',
      REPEAT ('0', 20)
    ),
    20 - (A.gender - 1) * 10 - (A.age - 1) + 1,
    20
  );
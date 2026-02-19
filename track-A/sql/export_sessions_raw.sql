/* export_sessions_raw.sql (wrapper)
   This file is NOT executed directly.
   It will be built by scripts/build_exports.sh by embedding sessions_query.sql into ( ... ) as subquery.
*/

SELECT *
FROM (
  /*__SESSIONS_QUERY__*/  -- 여기 자리에 sessions_query.sql이 들어감
)
FORMAT Parquet

/* export_session_features.sql (wrapper)
   This file is NOT executed directly.
*/

WITH sessions AS (
  /*__SESSIONS_QUERY__*/  -- 여기 자리에 sessions_query.sql이 들어감
)
SELECT
    user_id,
    client_name,
    session_key,
    session_start,
    session_end,
    n_events,

    dateDiff('second', session_start, session_end) AS duration_s,
    n_events / nullIf(duration_s + 1, 0) AS events_per_s,

    arrayCount(x -> x = 'error', outcomes) AS n_error,
    arrayCount(x -> x = 'rate_limited', outcomes) AS n_rate_limited,
    arrayCount(x -> x = 'auth_fail', outcomes) AS n_auth_fail,
    arrayCount(x -> x = 'guardrail_warn', outcomes) AS n_guardrail_warn,
    arrayCount(x -> x = 'guardrail_block', outcomes) AS n_guardrail_block,
    arrayCount(x -> x = 'bad_request', outcomes) AS n_bad_request,
    arrayCount(x -> x = 'internal_error', outcomes) AS n_internal_error,
    n_error / nullIf(n_events, 0) AS error_ratio,

    arrayCount(x -> x = 'chat', route_groups) AS n_chat,
    arrayCount(x -> x = 'embeddings', route_groups) AS n_embeddings,
    arrayCount(x -> x = 'models', route_groups) AS n_models,
    arrayCount(x -> x = 'images', route_groups) AS n_images,
    arrayCount(x -> x = 'audio', route_groups) AS n_audio,
    arrayCount(x -> x = 'files', route_groups) AS n_files,
    arrayCount(x -> x = 'moderations', route_groups) AS n_moderations,
    arrayCount(x -> x = 'batches', route_groups) AS n_batches,
    arrayCount(x -> x = 'assistants', route_groups) AS n_assistants,
    arrayCount(x -> x = 'inline', route_groups) AS n_inline,
    arrayCount(x -> x = 'other', route_groups) AS n_other,

    arrayCount(x -> x = 't_start', dt_buckets) AS n_t_start,
    arrayCount(x -> x = 't_fast', dt_buckets) AS n_t_fast,
    arrayCount(x -> x = 't_norm', dt_buckets) AS n_t_norm,
    arrayCount(x -> x = 't_slow', dt_buckets) AS n_t_slow,
    arrayCount(x -> x = 't_idle', dt_buckets) AS n_t_idle,
    n_t_fast / nullIf(n_events, 0) AS fast_ratio,
    n_t_idle / nullIf(n_events, 0) AS idle_ratio,

    length(arrayDistinct(tokens)) AS n_unique_tokens,
    n_unique_tokens / nullIf(n_events, 0) AS unique_token_ratio

FROM sessions
WHERE n_events >= 5
ORDER BY session_start
FORMAT Parquet

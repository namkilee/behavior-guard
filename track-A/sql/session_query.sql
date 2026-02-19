/* sessions_query.sql
   Placeholders:
   - {day_start:String}
   - {day_end:String}
*/

WITH
    toDateTime64({day_start:String}, 3) AS day_start,
    toDateTime64({day_end:String}, 3) AS day_end,
    30 AS gap_minutes,

    tr AS (
        SELECT
            project_id,
            id AS trace_id,
            coalesce(
                nullIf(user_id, ''),
                nullIf(metadata['user_api_key_user_id'], ''),
                nullIf(metadata['user_api_key_end_user_id'], '')
            ) AS user_id_norm,
            nullIf(session_id, '') AS session_id_norm,
            nullIf(metadata['aap_log_name'], '') AS trace_aap_log_name,
            nullIf(metadata['user_api_key_request_route'], '') AS trace_route_norm
        FROM traces
        WHERE created_at >= day_start AND created_at < day_end AND is_deleted = 0
    ),

    obs AS (
        SELECT
            project_id,
            id AS observation_id,
            trace_id,
            created_at AS event_time,
            name AS obs_name,
            level AS level_norm,
            metadata,
            coalesce(nullIf(metadata['aap_log_name'], ''), 'unknown_client') AS aap_log_name_obs,
            nullIf(metadata['user_api_key_request_route'], '') AS route_norm_obs
        FROM observations
        WHERE created_at >= day_start AND created_at < day_end AND is_deleted = 0
    ),

    base AS (
        SELECT
            o.event_time,
            o.project_id,
            o.observation_id,
            o.trace_id,
            o.obs_name,
            o.level_norm,
            o.metadata,
            t.user_id_norm,
            t.session_id_norm,
            coalesce(nullIf(o.aap_log_name_obs, 'unknown_client'), t.trace_aap_log_name, 'unknown_client') AS aap_log_name_norm,
            coalesce(nullIf(o.route_norm_obs, ''), t.trace_route_norm) AS route_norm
        FROM obs o
        LEFT JOIN tr t
            ON t.project_id = o.project_id AND t.trace_id = o.trace_id
    ),

    with_route_group AS (
        SELECT
            *,
            multiIf(
                route_norm IN ('/completions', '/v1/completions'), 'inline',
                route_norm IN ('/chat/completions', '/v1/chat/completions', '/responses', '/v1/responses'), 'chat',
                route_norm IN ('/embeddings', '/v1/embeddings'), 'embeddings',
                route_norm IN ('/models', '/v1/models'), 'models',
                startsWith(route_norm, '/images') OR startsWith(route_norm, '/v1/images'), 'images',
                startsWith(route_norm, '/audio') OR startsWith(route_norm, '/v1/audio'), 'audio',
                startsWith(route_norm, '/files') OR startsWith(route_norm, '/v1/files'), 'files',
                startsWith(route_norm, '/moderations') OR startsWith(route_norm, '/v1/moderations'), 'moderations',
                startsWith(route_norm, '/batches') OR startsWith(route_norm, '/v1/batches'), 'batches',
                (
                    startsWith(route_norm, '/assistants') OR startsWith(route_norm, '/v1/assistants')
                    OR startsWith(route_norm, '/threads') OR startsWith(route_norm, '/v1/threads')
                    OR startsWith(route_norm, '/runs') OR startsWith(route_norm, '/v1/runs')
                ), 'assistants',
                'other'
            ) AS route_group
        FROM base
    ),

    with_outcome AS (
        SELECT
            *,
            multiIf(
                obs_name = 'litellm-rate_limit_exceed', 'rate_limited',
                obs_name = 'litellm-auth_error', 'auth_fail',
                obs_name = 'litellm-guardrail_warning', 'guardrail_warn',
                obs_name = 'litellm-guardrail_violation', 'guardrail_block',
                obs_name = 'litellm-internal_error', 'internal_error',
                obs_name = 'litellm-bad_request_error', 'bad_request',
                obs_name = 'litellm-IGNORE_THIS', 'error',
                obs_name IN ('litellm-models', 'litellm-/v1/models', 'litellm-/models', 'litellm-info'), 'info',
                obs_name IN (
                    'litellm-acompletion', 'litellm-completion',
                    'litellm-completions', 'litellm-/chat/completions',
                    'litellm-aresponses'
                ), 'ok',
                'ok'
            ) AS outcome_seed,
            multiIf(
                outcome_seed IN ('rate_limited', 'auth_fail', 'guardrail_warn', 'guardrail_block',
                                 'internal_error', 'bad_request', 'error', 'info'), outcome_seed,
                (outcome_seed = 'ok' AND level_norm = 'ERROR'), 'error',
                'ok'
            ) AS outcome_class
        FROM with_route_group
    ),

    with_session_w0 AS (
        SELECT
            *,
            (session_id_norm IS NOT NULL) AS has_session_id,
            if(has_session_id, session_id_norm, 'NOSESSION') AS session_partition_key,
            lagInFrame(toNullable(event_time), 1, NULL) OVER (
                PARTITION BY user_id_norm, aap_log_name_norm, session_partition_key
                ORDER BY event_time
            ) AS prev_time_in_partition,
            row_number() OVER (
                PARTITION BY user_id_norm, aap_log_name_norm, session_partition_key
                ORDER BY event_time
            ) AS rn_in_partition
        FROM with_outcome
    ),

    with_session AS (
        SELECT
            *,
            multiIf(
                rn_in_partition = 1, 1,
                (NOT has_session_id)
                    AND prev_time_in_partition IS NOT NULL
                    AND dateDiff('minute', prev_time_in_partition, event_time) >= gap_minutes, 1,
                0
            ) AS session_start_marker,
            sum(session_start_marker) OVER (
                PARTITION BY user_id_norm, aap_log_name_norm, session_partition_key
                ORDER BY event_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS session_seq_in_partition,
            if(
                has_session_id,
                concat('sid:', session_id_norm),
                concat('gap:', toString(session_seq_in_partition))
            ) AS session_key
        FROM with_session_w0
    ),

    token_rows AS (
        SELECT
            trace_id,
            user_id_norm AS user_id,
            aap_log_name_norm AS client_name,
            session_key,
            event_time,
            route_group,
            obs_name AS raw_name,
            level_norm AS level,
            outcome_class,
            lagInFrame(toNullable(event_time), 1, NULL) OVER (
                PARTITION BY user_id_norm, aap_log_name_norm, session_key
                ORDER BY event_time
            ) AS prev_time_in_session,
            dateDiff('millisecond', prev_time_in_session, event_time) AS dt_ms,
            multiIf(
                prev_time_in_session IS NULL, 't_start',
                dt_ms < 200, 't_fast',
                dt_ms < 2000, 't_norm',
                dt_ms < 30000, 't_slow',
                't_idle'
            ) AS dt_bucket,
            concat(
                'client=', aap_log_name_norm,
                '|op=', route_group,
                '|out=', outcome_class,
                '|dt=', dt_bucket
            ) AS token
        FROM with_session
        WHERE user_id_norm IS NOT NULL AND user_id_norm != ''
    )

SELECT
    user_id,
    client_name,
    session_key,
    min(event_time) AS session_start,
    max(event_time) AS session_end,
    count() AS n_events,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, token)))) AS tokens,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, outcome_class)))) AS outcomes,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, route_group)))) AS route_groups,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, dt_bucket)))) AS dt_buckets,
    arraySort(groupArray(event_time)) AS event_times
FROM token_rows
GROUP BY user_id, client_name, session_key
ORDER BY session_start

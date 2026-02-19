WITH
    toDateTime('2026-01-06 00:00:00', 3) AS day_start,
    toDateTime('2026-01-07 00:00:00', 3) AS day_end,
    toDate('2026-01-06') AS day_key
SELECT
    user_id,
    aap_log_name,
    session_key,
    min(event_time) AS session_start,
    max(event_time) AS session_end,
    count() AS n_events,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, token)))) AS tokens,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, outcome_class)))) AS outcomes,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, route_group)))) AS route_groups,
    arrayMap(x -> x.2, arraySort(groupArray((event_time, token)))) AS event_times
FROM (
    SELECT
        t.user_id_norm as user_id,
        coalesce(nullIf(o.aap_log_name_obs, 'unknown_client'), t.trace_aap_log_name, 'unknown_client') AS aap_log_name,
        if(
            t.session_id_norm IS NOT NULL,
            concat('sid:', t.session_id_norm),
            concat('gap:', toString(day_key))
        ) AS session_key,
        o.event_time AS event_time,
        coalesce(nullIf(o.route_norm_obs, ''), t.trace_route_norm) AS route_norm,
        multiIf(
            route_norm IN ('/completions', '/v1/completions'), 'inline',
            route_norm IN ('/chat/completions', '/v1/chat/completions', '/responses', '/v1/responses'), 'chat',
            route_norm IN ('/embeddings', '/v1/embeddings'), 'embeddings',
            route_norm IN ('/models', '/v1/models'), 'models',
            startWith(route_norm, '/images') OR startWith(route_norm, '/v1/images'), 'images',
            startWith(route_norm, '/audio') OR startWith(route_norm, '/v1/audio'), 'audio',
            startWith(route_norm, '/files') OR startWith(route_norm, '/v1/files'), 'files',
            startWith(route_norm, '/moderations') OR startWith(route_norm, '/v1/moderations'), 'moderations',
            startWith(route_norm, '/batches') OR startWith(route_norm, '/v1/batches'), 'batches',
            (startWith(route_norm, '/assistants') OR startWith(route_norm, '/v1/assistants')
                OR startWith(route_norm, '/threads') OR startWith(route_norm, '/v1/threads')
                OR startWith(route_norm, '/runs') OR startWith(route_norm, '/v1/runs')
            ), 'assistants',
            'others'
        ) AS rounte_group,
        multiIf(
            obs_name = 'litellm-rate_limit_exceed', 'rate_limited',
            obs_name = 'litellm-auth_error', 'auth_fail',
            obs_name = 'litellm-guardrail_warning', 'guardrail_warn',
            obs_name = 'litellm-guardrail_violation', 'guardrail_block',
            obs_name = 'litellm-internal_error', 'internal_error',
            obs_name = 'litellm-bad_request_error', 'bad_request',
            obs_name = 'litellm-IGNORE_THIS', 'error',
            obs_name IN ('litellm-models', 'litellm-/v1/models', 'litellm-/models', 'litellm-info'), 'info',
            obs_name IN ('litellm-accompletion', 'litellm-completion',
                        'litellm-completions', 'litellm-/chat/completions',
                        'litellm-aresponses'), 'ok',
            'ok'
        ) AS outcome_seed,
        multiIf(
            outcome_seed IN ('rate_limited', 'auth_fail', 'guardrail_warn', 'guardrail_block',
                            'internal_error', 'bad_request', 'error', 'info'), outcome_seed,
            (outcome_seed = 'ok' AND level_norm = 'ERROR'), 'error',
            'ok'
        ) AS outcome_class,
        concat(
            'client=', coalesce(nullIf(o.aap_log_name_obs, 'unknown_client'), t.trace_aap_log_name, 'unknown_client'),
            '|op=', route_group,
            '|out=', outcome_class,
            '|dt=t_na'
        ) AS token
    FROM (
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
        FROM observation
        WHERE created_at >= day_start AND created_at < day_end AND is_deleted = 0
    ) o
    LEFT JOIN (
        SELECT
            project_id,
            id AS trace_id,
            coalesce(nullIf(user_id, ''),
                    nullIf(metadata['user_api_key_user_id'], ''),
                    nullIf(metadata['user_api_key_end_user_id'], '')
            ) AS user_id_norm
            nullIf(session_id, '') AS session_id_norm,
            nullIf(metadata['aap_log_name'], '') AS trace_aap_log_name,
            nullIf(metadata['user_api_key_request_route'], '') AS trace_route_norm
        FROM traces
        WHERE created_at >= day_start AND created_at < day_end AND is_deleted = 0
    ) t
    ON t.project_id = o.project_id AND t.trace_id = o.trace_id
    WHERE t.user_id_norm IS NOT NULL AND t.user_id_norm != ''
) event_rows
GROUP BY user_id, aap_log_name, session_key
ORDER BY user_id, aap_log_name, session_start;
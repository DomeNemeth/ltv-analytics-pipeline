-- How much the recent-run-rate baseline depends on the length of its look-back window.
--
-- The scored model uses one window (var recent_window_days, 91 days). Picking that window after
-- seeing which one beats the model would be cherry-picking, and a reader cannot tell a tuned
-- window from a principled one by looking at a single number. So the same rule is also totalled
-- at roughly one, two and three months. If the conclusion changed across these, the report would
-- have to say so.
--
-- Aggregate only, one row per (source, window_days). Per-customer scoring stays in
-- int_customers__scored, for the configured window alone.

{% set windows = [30, 61, 91] %}
{% if var('recent_window_days') | int not in windows %}
    {% do windows.append(var('recent_window_days') | int) %}
{% endif %}

with recent as (

    {% for window_days in windows %}
    {{ recent_repeats(window_days) }}
    {% if not loop.last %}union all{% endif %}
    {% endfor %}

),

windows as (

    select * from {{ ref('int_sources__analysis_windows') }}

),

predicted as (

    select
        recent.source,
        recent.window_days,
        count(*) as customers,
        sum(
            case
                when recent.recent_exposure_days > 0
                    then cast(recent.recent_repeats as double)
                        / recent.recent_exposure_days
                        * windows.duration_holdout
                else 0.0
            end
        ) as predicted_purchases

    from recent
    inner join windows
        on recent.source = windows.source
    group by recent.source, recent.window_days

),

actual as (

    select
        source,
        sum(holdout_frequency) as actual_purchases
    from {{ ref('int_customers__holdout_actuals') }}
    group by source

)

select
    predicted.source,
    predicted.window_days,
    predicted.customers,
    predicted.predicted_purchases,
    actual.actual_purchases,
    100.0 * (predicted.predicted_purchases - actual.actual_purchases)
        / actual.actual_purchases as percent_error

from predicted
inner join actual
    on predicted.source = actual.source

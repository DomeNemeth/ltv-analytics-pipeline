-- How much the recent-run-rate baseline depends on the length of its look-back window.
--
-- The scored model uses one window (var recent_window_days, 91 days). A single number cannot show
-- whether a window was tuned, so the same rule is also totalled at one to nine months. The
-- first version stopped at three months and claimed the conclusion did not depend on the window.
-- The Phase 4 re-audit extended it: from about four months back the rule loses to the model. The
-- range now runs out to the full calibration length, where the rule meets the calibration-rate
-- baseline, so the report can compute where the crossover falls instead of asserting it.
--
-- Aggregate only, one row per (source, window_days). Per-customer scoring stays in
-- int_customers__scored, for the configured window alone.
-- assert_recent_window_sensitivity_matches_the_scored_baseline ties the two together.

{% set windows = [30, 61, 91, 122, 152, 182, 273] %}
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

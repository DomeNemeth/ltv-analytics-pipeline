{{ config(tags=['post_fit']) }}

-- The sensitivity row for the configured window equals the per-customer baseline, summed.
--
-- The report's sentence about how the recent-rate rule depends on its window is computed from
-- int_baselines__recent_window_sensitivity. Nothing checked those totals against anything. The
-- Phase 4 re-audit multiplied them by 1.5 on a copy of the warehouse and every guard stayed green,
-- so the sentence could have said anything. The configured window is the one row that has an
-- independent counterpart: int_customers__scored.baseline_recent, scored customer by customer
-- and itself recomputed by assert_baselines_use_only_calibration_inputs. The two sides share
-- the recent_repeats macro, so this does not check the macro. It checks the aggregation built
-- on top of it, which is what the report reads.

with sensitivity as (

    select source, predicted_purchases, actual_purchases
    from {{ ref('int_baselines__recent_window_sensitivity') }}
    where window_days = {{ var('recent_window_days') }}

),

scored as (

    select
        source,
        sum(baseline_recent) as predicted_purchases,
        sum(holdout_frequency) as actual_purchases
    from {{ ref('int_customers__scored') }}
    group by source

)

select
    scored.source,
    sensitivity.predicted_purchases as sensitivity_predicted,
    scored.predicted_purchases as scored_predicted,
    sensitivity.actual_purchases as sensitivity_actual,
    scored.actual_purchases as scored_actual

from scored

-- A left join, so a missing sensitivity row for the configured window fails too.
left join sensitivity
    on scored.source = sensitivity.source

where sensitivity.source is null
    or abs(sensitivity.predicted_purchases - scored.predicted_purchases)
        > 1e-9 * abs(scored.predicted_purchases)
    or sensitivity.actual_purchases != scored.actual_purchases

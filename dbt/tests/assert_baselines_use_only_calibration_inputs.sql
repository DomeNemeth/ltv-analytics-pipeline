{{ config(tags=['post_fit']) }}

-- The naive baselines are functions of calibration-window columns and nothing else.
--
-- A baseline exists to answer "did the model beat doing something obvious". That only works if the
-- obvious thing is available at prediction time. A baseline that reached for holdout_frequency or
-- holdout_spend would be reading the answer, and the honest conclusion -- "the model does not beat a
-- naive rule" -- would be drawn from a rule that cheated. Nothing else in the project would object:
-- the numbers stay positive, ordered sensibly, and roughly the right size.
--
-- Recomputed here from the calibration relation directly, with no reference to
-- int_customers__scored's own arithmetic. That does not make the two derivations independent in
-- origin -- they are the same formulae written twice -- and the test does not claim it does. What it
-- claims is narrower and is the thing that actually needs guarding: if the baselines in the model
-- are ever edited, this fails unless the same edit is made against calibration-only inputs. Reaching
-- for a holdout column cannot be done twice, because this side has none in scope.

with calibration as (

    select * from {{ ref('int_customers__rfm_calibration') }}

),

windows as (

    select * from {{ ref('int_sources__analysis_windows') }}

),

population as (

    select
        source,
        cast(sum(frequency) as double) / sum(customer_age) as repeat_rate_per_day,
        cast(
            avg(case when is_gamma_gamma_eligible then monetary_value end) as double
        ) as mean_repeat_order_value

    from calibration
    group by source

),

recomputed as (

    select
        calibration.source,
        calibration.customer_id,

        case
            when calibration.customer_age > 0
                then cast(calibration.frequency as double)
                    / calibration.customer_age
                    * windows.duration_holdout
            else 0.0
        end as baseline_purchases,

        population.repeat_rate_per_day * windows.duration_holdout as baseline_purchases_flat,

        case
            when calibration.is_gamma_gamma_eligible
                then cast(calibration.monetary_value as double)
            else population.mean_repeat_order_value
        end as baseline_avg_value,

        population.repeat_rate_per_day
        * windows.duration_holdout
        * population.mean_repeat_order_value as baseline_revenue_flat,

        cast(calibration.frequency as double) as baseline_carry_forward,

        cast(0.0 as double) as baseline_zero

    from calibration

    inner join windows
        on calibration.source = windows.source

    inner join population
        on calibration.source = population.source

)

select
    scored.source,
    scored.customer_id,
    scored.baseline_purchases,
    recomputed.baseline_purchases as expected_baseline_purchases,
    scored.baseline_avg_value,
    recomputed.baseline_avg_value as expected_baseline_avg_value

from {{ ref('int_customers__scored') }} as scored

inner join recomputed
    on scored.source = recomputed.source
    and scored.customer_id = recomputed.customer_id

-- Every baseline column the report quotes is checked here. `baseline_revenue_flat` was reported as
-- +33.0% for a while with no description, no not_null, and no line in this test -- the same shape
-- as the Phase 2 finding where holdout money was emitted and never guarded. A number that appears
-- in a report and in no test is a number nobody has checked.
--
-- Tolerance is for floating-point association only. Any real change to a baseline moves it by
-- orders of magnitude more than this.
where abs(scored.baseline_purchases - recomputed.baseline_purchases) > 1e-9
    or abs(scored.baseline_purchases_flat - recomputed.baseline_purchases_flat) > 1e-9
    or abs(scored.baseline_avg_value - recomputed.baseline_avg_value) > 1e-9
    or abs(scored.baseline_revenue_flat - recomputed.baseline_revenue_flat) > 1e-9
    or abs(scored.baseline_carry_forward - recomputed.baseline_carry_forward) > 1e-9
    or abs(scored.baseline_zero - recomputed.baseline_zero) > 1e-9
    or abs(
        scored.baseline_revenue - recomputed.baseline_purchases * recomputed.baseline_avg_value
    ) > 1e-9
    or abs(
        scored.baseline_revenue_carry_forward
        - recomputed.baseline_carry_forward * recomputed.baseline_avg_value
    ) > 1e-9

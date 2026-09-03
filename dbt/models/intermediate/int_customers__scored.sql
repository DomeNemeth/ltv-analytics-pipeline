{{ config(materialized='table', tags=['post_fit']) }}

-- Everything needed to judge a prediction, on one row per customer: what the model said, what the
-- customer actually did, what a naive rule would have said instead, and the calibration features
-- all three are conditioned on.
--
-- This is the relation `ltv validate` computes its metrics from, and the one the Phase 5 marts will
-- read. It exists in dbt rather than in pandas for a specific reason: the two worst defects this
-- project has shipped were both in a join and both invisible to the code that consumed it. A join
-- here can be tested; a join inside a metrics function cannot.
--
-- Three things about it are load-bearing:
--
-- 1. **The horizon is derived, never literal.** Predictions are joined on
--    `horizon_days = duration_holdout`, so the forecast being scored always covers exactly the
--    window it is scored against. A hardcoded 273 here would keep working until the split changed
--    and then quietly compare a 365-day forecast to a 273-day outcome.
--
-- 2. **Every join out of the calibration population is a left join, and the population itself is
--    never filtered.** The 16,512 CDNOW customers who never returned are the ones the model
--    predicts worst, so dropping them improves every metric dramatically. The filtering is the
--    real risk here rather than the join form: assert_customer_populations_align already pins the
--    holdout and calibration populations to the same keys, so `inner join actuals` would behave
--    identically today -- mutation testing confirmed it, which is why the harness mutates a
--    `where` clause on the calibration CTE instead. The left joins stay because that upstream
--    guarantee is not this model's to make, and because a missing row must survive to be counted:
--    hence not_null on holdout_frequency and expected_purchases rather than a coalesce, since a
--    zero and a missing row mean entirely different things.
--
-- 3. **The baselines are functions of calibration columns only.** A baseline that reached for a
--    holdout column would beat the model by cheating, and the model would look worse than nothing.
--    assert_baselines_use_only_calibration_inputs recomputes all three independently.
--
-- Note the scoring asymmetry, which is correct: holdout_frequency counts *every* occasion while
-- calibration frequency counts repeats only. BG/NBD's expected_purchases predicts all purchases in
-- the forward window -- there is no "first purchase" to exclude in a window that has not happened
-- yet -- so total against total is the right comparison. See int_customers__holdout_actuals.

with calibration as (

    select * from {{ ref('int_customers__rfm_calibration') }}

),

actuals as (

    select * from {{ ref('int_customers__holdout_actuals') }}

),

windows as (

    select * from {{ ref('int_sources__analysis_windows') }}

),

predictions as (

    select * from {{ ref('stg_model__customer_predictions') }}

),

population as (

    -- The constants behind the "everyone is average" baseline, computed over the calibration window
    -- only. Repeat occasions per customer-day across the whole population, and the mean repeat order
    -- value among customers who have one -- the same quantity Gamma-Gamma falls back to for
    -- customers it cannot fit, so the floor and the model treat that group alike.
    select
        source,
        cast(sum(frequency) as double) / sum(customer_age) as repeat_rate_per_day,
        cast(
            avg(case when is_gamma_gamma_eligible then monetary_value end) as double
        ) as mean_repeat_order_value

    from calibration
    group by source

)

select
    calibration.source,
    calibration.customer_id,

    -- Calibration features, carried so metrics can be cut by them without another join. The
    -- frequency buckets in the validation report are built from `frequency`.
    calibration.frequency,
    calibration.recency,
    calibration.customer_age,
    calibration.monetary_value,
    calibration.is_gamma_gamma_eligible,

    -- The scoring window.
    windows.duration_holdout,
    predictions.horizon_days,

    -- Ground truth.
    actuals.holdout_frequency,
    actuals.holdout_spend,
    actuals.holdout_monetary_value,

    -- What the model said.
    predictions.expected_purchases,
    predictions.probability_alive,
    predictions.expected_avg_value,
    predictions.expected_forward_revenue,
    predictions.spend_estimate_source,
    predictions.fit_method,

    -- Naive baseline 1: this customer keeps repeating at the rate they established during
    -- calibration. The one that is genuinely hard to beat, and therefore the one worth reporting.
    -- Guarded against a zero customer_age, which cannot happen on CDNOW but can the moment a source
    -- with a later acquisition tail lands.
    case
        when calibration.customer_age > 0
            then cast(calibration.frequency as double)
                / calibration.customer_age
                * windows.duration_holdout
        else 0.0
    end as baseline_purchases,

    -- Naive baseline 2: everyone repeats at the population rate. The floor -- it uses no
    -- customer-level information at all, so a model that cannot beat it has learned nothing.
    population.repeat_rate_per_day * windows.duration_holdout as baseline_purchases_flat,

    -- Past average order value is future average order value.
    case
        when calibration.is_gamma_gamma_eligible then cast(calibration.monetary_value as double)
        else population.mean_repeat_order_value
    end as baseline_avg_value,

    case
        when calibration.customer_age > 0
            then cast(calibration.frequency as double)
                / calibration.customer_age
                * windows.duration_holdout
        else 0.0
    end
    * case
        when calibration.is_gamma_gamma_eligible then cast(calibration.monetary_value as double)
        else population.mean_repeat_order_value
    end as baseline_revenue,

    population.repeat_rate_per_day
    * windows.duration_holdout
    * population.mean_repeat_order_value as baseline_revenue_flat,

    -- Naive baseline 3: whatever you did last window, you will do again. Added after a validation
    -- audit pointed out that the two baselines above share a hidden inflation, and that it was
    -- flattering the model by roughly half of the reported margin.
    --
    -- `baseline_purchases` divides by each customer's own observation length and multiplies by the
    -- holdout length. That is a correct Poisson-rate estimate, but mean customer_age on CDNOW is
    -- 229 days against a 273-day holdout, so it scales every calibration count up by 273/229 =
    -- 1.19 before comparing. The resulting "+47%" is therefore partly the rescaling rather than
    -- the naivety, and both rate baselines inherit it because it is one factor applied two ways.
    --
    -- Here the two windows are the same length -- 39 weeks each, to the day -- so the honest
    -- same-period rule needs no rescaling at all and is the harder thing to beat. Reported
    -- alongside the others rather than replacing them: the rate baseline is the right rule when
    -- windows differ, which they will the moment a second source lands.
    cast(calibration.frequency as double) as baseline_carry_forward,

    cast(calibration.frequency as double)
    * case
        when calibration.is_gamma_gamma_eligible then cast(calibration.monetary_value as double)
        else population.mean_repeat_order_value
    end as baseline_revenue_carry_forward,

    -- The floor beneath every floor: predict that nobody buys anything. Not a serious rule, and
    -- that is the point. On a quantity where 70% of the actuals are zero, this is what per-customer
    -- MAE is actually measured against -- a model that does not clearly beat it has not shown any
    -- individual-level skill, whatever its aggregate error says. Reported so the MAE column cannot
    -- be read as accuracy without meeting it.
    0.0 as baseline_zero

from calibration

inner join windows
    on calibration.source = windows.source

inner join population
    on calibration.source = population.source

left join actuals
    on calibration.source = actuals.source
    and calibration.customer_id = actuals.customer_id

left join predictions
    on calibration.source = predictions.source
    and calibration.customer_id = predictions.customer_id
    and predictions.horizon_days = windows.duration_holdout

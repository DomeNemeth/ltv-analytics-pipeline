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
--    assert_baselines_use_only_calibration_inputs recomputes every one of them independently.
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

recent_repeats as (

    {{ recent_repeats(var('recent_window_days')) }}

),

recent as (

    -- The recent repeat rate per day, computed once so the purchase and revenue baselines below
    -- cannot disagree about it.
    select
        source,
        customer_id,
        window_days,
        case
            when recent_exposure_days > 0
                then cast(recent_repeats as double) / recent_exposure_days
            else 0.0
        end as repeats_per_day
    from recent_repeats

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
    -- NULL on every MAP row by construction. Carried so the report can compute what the
    -- intervals cover from the table itself, rather than quoting a figure from one past run.
    predictions.forward_revenue_hdi_low,
    predictions.forward_revenue_hdi_high,
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

    -- Naive baseline 3: whatever you did last window, you will do again.
    --
    -- An earlier version of this comment called the two rate baselines above "inflated". The
    -- Phase 4 audit showed that reading was backwards. `baseline_purchases` divides by each
    -- customer's own exposure and multiplies by the holdout length, which is a correct
    -- exposure adjustment. This rule is the biased one: on CDNOW a customer was observed for 229
    -- days of calibration on average, not 273, so it under-counts exposure by about a sixth. It
    -- lands closer to the actual total only because that under-count partly cancels the rate
    -- rules' real problem: the purchase rate fell through calibration, so any average over the
    -- whole window overshoots the holdout. Two errors cancelling are not a fairer comparison. They
    -- are a luckier one.
    --
    -- Kept because it is the rule a business would most likely use, and because the report has to
    -- show where each naive rule's number comes from.
    cast(calibration.frequency as double) as baseline_carry_forward,

    cast(calibration.frequency as double)
    * case
        when calibration.is_gamma_gamma_eligible then cast(calibration.monetary_value as double)
        else population.mean_repeat_order_value
    end as baseline_revenue_carry_forward,

    -- Naive baseline 4: this customer keeps repeating at the rate of their last quarter. Added after
    -- the Phase 4 audit found it closer on the total than the model. Every rule above averages
    -- over the whole calibration window, but the purchase rate falls steeply through that window,
    -- so they all overshoot. A rule that looks only at the end of the window does not.
    -- This is the non-stationarity the report diagnoses, and it is not a model's to exploit, but
    -- it is available to anyone with the data, so it is the comparison the model has to face.
    --
    -- The window is fixed at 91 days (one quarter) on conventional grounds, not on results. The
    -- auditor measured 30, 61 and 91 days, and 91 is the least flattering to the rule of the
    -- three. int_baselines__recent_window_sensitivity reports all three so a reader can see the
    -- choice does not carry the conclusion.
    recent.window_days as recent_window_days,

    -- No coalesce. The macro keeps every calibration customer, so a NULL here means a row went
    -- missing in the join, and the not_null test must see it rather than a quiet zero.
    recent.repeats_per_day * windows.duration_holdout as baseline_recent,

    recent.repeats_per_day
    * windows.duration_holdout
    * case
        when calibration.is_gamma_gamma_eligible then cast(calibration.monetary_value as double)
        else population.mean_repeat_order_value
    end as baseline_revenue_recent,

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

left join recent
    on calibration.source = recent.source
    and calibration.customer_id = recent.customer_id

left join predictions
    on calibration.source = predictions.source
    and calibration.customer_id = predictions.customer_id
    and predictions.horizon_days = windows.duration_holdout

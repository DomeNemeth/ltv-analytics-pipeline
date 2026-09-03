{{ config(tags=['post_fit']) }}

-- The predictions in the warehouse were fitted on the data that is in the warehouse now.
--
-- This is the guard for a failure mode that has already happened here once and left no trace in
-- `git diff`. A mutation-testing run restored every source file it touched but not the warehouse
-- those files had materialised, so int_customers__rfm_calibration was left holding monetary_value
-- divided by the wrong denominator -- about a third too low. The next `ltv fit` read it, trained
-- Gamma-Gamma on corrupted spend, and wrote a predictions table that looked entirely normal:
-- positive, smaller than total spend, correctly zero for one-time buyers.
--
-- The general shape of it is broader than that one bug. `ltv transform` can be re-run at any time
-- after `ltv fit`, and a rebuilt calibration relation leaves the predictions stale with nothing to
-- say so. Phase 4 scores those predictions, so "stale" means "every metric in the report is wrong".
--
-- `ltv fit` records what it actually trained on -- summary statistics of the in-memory frame, not of
-- the relation it hoped to read. This recomputes them from the relation as it stands and fails on
-- any disagreement. Sums rather than a hash so a failure names the quantity that moved.

with recorded as (

    select * from {{ source('model', 'fit_runs') }}

),

current_inputs as (

    select
        source,
        count(*) as calibration_rows,
        sum(frequency) as sum_frequency,
        sum(recency) as sum_recency,
        sum(customer_age) as sum_customer_age,
        cast(sum(monetary_value) as double) as sum_monetary_value

    from {{ ref('int_customers__rfm_calibration') }}
    group by source

)

select
    current_inputs.source,
    recorded.fitted_at,
    recorded.fit_method,
    recorded.calibration_rows as fitted_on_rows,
    current_inputs.calibration_rows as warehouse_rows,
    recorded.sum_monetary_value as fitted_on_spend,
    current_inputs.sum_monetary_value as warehouse_spend

from current_inputs

-- A left join, so a calibration population with no fit_runs row at all fails this rather than
-- disappearing from the comparison. That is the state a warehouse is in when `ltv transform` has
-- run and `ltv fit` has not, which is exactly when scoring must refuse to proceed.
left join recorded
    on current_inputs.source = recorded.source

where recorded.source is null
    or recorded.calibration_rows != current_inputs.calibration_rows
    or recorded.sum_frequency != current_inputs.sum_frequency
    or recorded.sum_recency != current_inputs.sum_recency
    or recorded.sum_customer_age != current_inputs.sum_customer_age
    -- Tolerance covers the difference between a float64 sum in pandas and a decimal sum in DuckDB
    -- over 23,570 rows, which is around 1e-9. Any real change to spend is cents at minimum.
    or abs(recorded.sum_monetary_value - current_inputs.sum_monetary_value) > 0.01

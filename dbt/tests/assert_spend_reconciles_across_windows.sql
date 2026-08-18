-- Every dollar a customer spent falls in exactly one window: calibration spend plus holdout spend
-- must equal full-period spend, per customer.
--
-- The occasion reconciliation next door proves this for *counts*. Nothing proved it for money, and
-- money is what Gamma-Gamma is fitted on and what the dashboard reports. Two plausible corruptions
-- of holdout_spend -- summing line_items instead of gross_amount, and halving it -- survived all 62
-- tests before this one existed, because no test read the column at all.
--
-- The identity holds exactly today across all 23,570 CDNOW customers. Exact equality rather than a
-- cent tolerance is deliberate: source amounts are two-decimal and the collapse only ever sums
-- them, so no rounding can enter. If Phase 7 introduces a source whose gross_amount is derived
-- (quantity * unit_price at higher precision), this test will say so rather than silently absorbing
-- the drift, and the tolerance decision gets made in the open.

with reconciliation as (

    select
        calibration.source,
        calibration.customer_id,
        calibration.total_spend as calibration_spend,
        holdout.holdout_spend,
        full_period.total_spend as full_spend

    from {{ ref('int_customers__rfm_calibration') }} as calibration

    inner join {{ ref('int_customers__rfm_full') }} as full_period
        on calibration.source = full_period.source
        and calibration.customer_id = full_period.customer_id

    inner join {{ ref('int_customers__holdout_actuals') }} as holdout
        on calibration.source = holdout.source
        and calibration.customer_id = holdout.customer_id

)

select *
from reconciliation
where calibration_spend + holdout_spend != full_spend

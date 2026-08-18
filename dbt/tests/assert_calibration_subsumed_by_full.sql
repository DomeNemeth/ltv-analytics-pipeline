-- The calibration window is a prefix of the full window, so every calibration quantity must be
-- bounded by its full-period counterpart.
--
-- This is what catches the two windows drifting out of alignment -- a changed cutoff applied to one
-- model and not the other, or a `between` that became exclusive on one side. Phase 4 compares these
-- two models directly, so a misalignment here would show up as model error rather than as a bug.

select
    calibration.source,
    calibration.customer_id,
    calibration.frequency as calibration_frequency,
    full_period.frequency as full_frequency,
    calibration.total_spend as calibration_spend,
    full_period.total_spend as full_spend

from {{ ref('int_customers__rfm_calibration') }} as calibration

inner join {{ ref('int_customers__rfm_full') }} as full_period
    on calibration.source = full_period.source
    and calibration.customer_id = full_period.customer_id

where
    calibration.frequency > full_period.frequency
    or calibration.total_spend > full_period.total_spend
    or calibration.customer_age > full_period.customer_age

    -- Both windows open on the same day, so the first purchase must be identical. If it is not, one
    -- of the windows is starting somewhere else.
    or calibration.first_order_date != full_period.first_order_date

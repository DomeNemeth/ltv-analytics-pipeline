-- Calibration occasions plus holdout occasions must account for every occasion a customer had.
--
-- If they do not, the two windows either overlap (double-counting purchases, and leaking holdout
-- behaviour into the training features) or leave a gap (silently discarding purchases). Both look
-- entirely normal in the output tables.
--
-- The joins are full outer rather than inner on purpose. With inner joins this test examined only
-- the customers present in all three models, so a defect that *removed* customers was invisible to
-- it -- the rows that would have failed were the rows the join had already dropped. That is how a
-- regression turning the left join in int_customers__holdout_actuals into an inner join, losing
-- 16,512 of 23,570 customers, passed the entire suite.

with reconciliation as (

    select
        coalesce(calibration.source, full_period.source, holdout.source) as source,
        coalesce(
            calibration.customer_id, full_period.customer_id, holdout.customer_id
        ) as customer_id,
        calibration.occasions as calibration_occasions,
        holdout.holdout_frequency,
        full_period.occasions as full_occasions

    from {{ ref('int_customers__rfm_calibration') }} as calibration

    full outer join {{ ref('int_customers__rfm_full') }} as full_period
        on calibration.source = full_period.source
        and calibration.customer_id = full_period.customer_id

    full outer join {{ ref('int_customers__holdout_actuals') }} as holdout
        on coalesce(calibration.source, full_period.source) = holdout.source
        and coalesce(calibration.customer_id, full_period.customer_id) = holdout.customer_id

)

select *
from reconciliation
where
    -- Present in one model and missing from another. assert_customer_populations_align states this
    -- more directly; it is repeated here so the reconciliation below can never be reading a
    -- silently truncated population and reporting success.
    calibration_occasions is null
    or holdout_frequency is null
    or full_occasions is null

    or calibration_occasions + holdout_frequency != full_occasions

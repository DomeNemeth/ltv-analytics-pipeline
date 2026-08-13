-- Calibration occasions plus holdout occasions must account for every occasion a customer had.
--
-- If they do not, the two windows either overlap (double-counting purchases, and leaking holdout
-- behaviour into the training features) or leave a gap (silently discarding purchases). Both look
-- entirely normal in the output tables.

with reconciliation as (

    select
        calibration.source,
        calibration.customer_id,
        calibration.occasions as calibration_occasions,
        holdout.holdout_frequency,
        full_period.occasions as full_occasions

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
where calibration_occasions + holdout_frequency != full_occasions

{{ config(tags=['post_fit']) }}

-- Every calibration customer is scored exactly once, and nothing else is.
--
-- The failure this exists for is not hypothetical and not subtle in its effect, only in its
-- appearance. Turning any of int_customers__scored's left joins into an inner join drops the
-- customers with no holdout row or no prediction row -- on CDNOW, the 16,512 who never came back,
-- which is 70% of the population and precisely the group the model predicts worst. Every error
-- metric in the project would improve sharply and nothing else would look different. The identical
-- defect in int_customers__holdout_actuals passed all 62 tests in Phase 2.
--
-- Counted per side rather than checked for presence, so a duplicated row fails this too. A fan-out
-- in the predictions join -- two horizons matching instead of one -- would otherwise double the
-- population while leaving every customer present.

with sides as (

    select 'calibration' as side, source, customer_id
    from {{ ref('int_customers__rfm_calibration') }}

    union all

    select 'scored' as side, source, customer_id
    from {{ ref('int_customers__scored') }}

)

select
    source,
    customer_id,
    count(*) filter (where side = 'calibration') as calibration_rows,
    count(*) filter (where side = 'scored') as scored_rows

from sides
group by source, customer_id
having count(*) filter (where side = 'calibration') != 1
    or count(*) filter (where side = 'scored') != 1

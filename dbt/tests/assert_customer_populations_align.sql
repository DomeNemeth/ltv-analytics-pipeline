-- The three per-customer summaries must describe exactly the same population.
--
-- This replaces two `relationships` tests that looked like they asserted it and did not. Both were
-- defective in a way reading could not reveal:
--
-- 1. int_customers__holdout_actuals selects customer_id *out of* int_customers__rfm_calibration, so
--    a relationship from the former to the latter holds by construction. No reachable defect can
--    violate it. It was decorative on the one model Phase 4 scores every prediction against.
-- 2. dbt's `relationships` can only key on a single column, and CLAUDE.md section 5 forbids keying
--    on customer_id alone: numbering is unique only within a dataset, so in Phase 7 an Online Retail
--    customer would satisfy the test by matching an unrelated CDNOW customer of the same number.
--
-- What actually needs asserting is equality, not containment, and the reason is concrete. The
-- `left join` in int_customers__holdout_actuals is what keeps the 16,512 CDNOW customers who never
-- returned in the comparison with a zero. Turning it into an inner join drops the population from
-- 23,570 to 7,058 and every holdout metric improves dramatically, because the customers the model
-- got wrong are the ones that disappear. That regression passed all 62 tests before this one
-- existed.
--
-- Combined with assert_one_customer_row_per_source, which forbids duplicates, this pins the three
-- populations to the same set of (source, customer_id) keys.

with populations as (

    select 'int_customers__rfm_calibration' as model_name, source, customer_id
    from {{ ref('int_customers__rfm_calibration') }}

    union all

    select 'int_customers__rfm_full' as model_name, source, customer_id
    from {{ ref('int_customers__rfm_full') }}

    union all

    select 'int_customers__holdout_actuals' as model_name, source, customer_id
    from {{ ref('int_customers__holdout_actuals') }}

),

coverage as (

    select
        source,
        customer_id,
        count(distinct model_name) as models_present,
        string_agg(distinct model_name, ', ' order by model_name) as present_in

    from populations
    group by source, customer_id

)

select
    source,
    customer_id,
    models_present,
    present_in

from coverage
where models_present != 3

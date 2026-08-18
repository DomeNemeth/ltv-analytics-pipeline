-- The per-customer summaries must hold exactly one row per customer per source.
--
-- A plain `unique` test on customer_id would pass today and start failing the moment a second
-- source lands, because customer numbering is only unique within a dataset. This states the real
-- grain instead, so it keeps meaning the same thing in Phase 7.

with all_summaries as (

    select 'int_customers__rfm_calibration' as model_name, source, customer_id
    from {{ ref('int_customers__rfm_calibration') }}

    union all

    select 'int_customers__rfm_full' as model_name, source, customer_id
    from {{ ref('int_customers__rfm_full') }}

    union all

    select 'int_customers__holdout_actuals' as model_name, source, customer_id
    from {{ ref('int_customers__holdout_actuals') }}

)

select
    model_name,
    source,
    customer_id,
    count(*) as rows_for_customer

from all_summaries
group by model_name, source, customer_id
having count(*) > 1

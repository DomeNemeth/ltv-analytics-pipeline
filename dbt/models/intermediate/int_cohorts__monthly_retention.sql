{{ config(materialized='table') }}

-- Monthly acquisition-cohort retention: of the customers first seen in month M, how many bought
-- again in each later month.
--
-- Cohorts are assigned from first purchase over the FULL history, not the calibration window, and
-- that is not a leakage violation. CLAUDE.md section 6 forbids post-cutoff dates in *calibration
-- features* -- things fed to the model. This is a descriptive report that no model consumes.
--
-- On CDNOW this yields exactly three cohorts (January, February, March 1997), because the dataset
-- is by construction the Q1-1997 acquisition cohort. That is a property of the data, not of this
-- model: Online Retail produces around thirteen from the same SQL in Phase 7.

with occasions as (

    select * from {{ ref('int_customers__purchase_occasions') }}

),

first_purchase as (

    select
        source,
        customer_id,
        cast(date_trunc('month', min(order_date)) as date) as cohort_month
    from occasions
    group by source, customer_id

),

cohort_sizes as (

    select
        source,
        cohort_month,
        count(*) as cohort_size
    from first_purchase
    group by source, cohort_month

),

calendar as (

    -- Every month in which anything happened, used to give each cohort a complete row of months
    -- rather than only the ones where it happened to be active. A heatmap with holes in it reads as
    -- missing data; a heatmap with explicit zeros reads as churn.
    select distinct
        source,
        cast(date_trunc('month', order_date) as date) as activity_month
    from occasions

),

grid as (

    select
        cohort_sizes.source,
        cohort_sizes.cohort_month,
        cohort_sizes.cohort_size,
        calendar.activity_month,
        date_diff('month', cohort_sizes.cohort_month, calendar.activity_month)
            as months_since_first
    from cohort_sizes
    inner join calendar
        on cohort_sizes.source = calendar.source
    where calendar.activity_month >= cohort_sizes.cohort_month

),

activity as (

    select
        occasions.source,
        first_purchase.cohort_month,
        cast(date_trunc('month', occasions.order_date) as date) as activity_month,
        count(distinct occasions.customer_id) as active_customers
    from occasions
    inner join first_purchase
        on occasions.source = first_purchase.source
        and occasions.customer_id = first_purchase.customer_id
    group by 1, 2, 3

)

select
    grid.source,
    grid.cohort_month,
    grid.months_since_first,
    grid.activity_month,
    grid.cohort_size,
    coalesce(activity.active_customers, 0) as active_customers,

    -- Month 0 is always 1.0 by construction: every customer is active in the month they are
    -- acquired. Kept rather than suppressed so the heatmap has its anchor column.
    cast(
        coalesce(activity.active_customers, 0) / grid.cohort_size as decimal(6, 4)
    ) as retention_rate

from grid
left join activity
    on grid.source = activity.source
    and grid.cohort_month = activity.cohort_month
    and grid.activity_month = activity.activity_month

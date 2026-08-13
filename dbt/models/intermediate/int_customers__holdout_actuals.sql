-- What each customer actually did in the holdout window. This is the ground truth Phase 4 scores
-- predictions against, so it is the one model that is allowed to look past the calibration cutoff.
--
-- Note the asymmetry with int_customers__rfm_calibration, which is correct and deliberate:
-- calibration `frequency` excludes the first purchase because BG/NBD models repeat behaviour,
-- whereas `holdout_frequency` counts every occasion in the window, because the model's prediction
-- is "how many purchases will this customer make next" -- all of them, not all-but-one. Scoring a
-- repeat count against a total count would understate the model's error by exactly one purchase
-- per active customer.

with customers as (

    -- Driven from the calibration population, so a customer who bought nothing in the holdout
    -- window still appears with a zero rather than dropping out of the comparison entirely. On
    -- CDNOW that is 16,512 of 23,570 customers -- silently losing them would flatter every metric.
    select source, customer_id from {{ ref('int_customers__rfm_calibration') }}

),

windows as (

    select * from {{ ref('int_sources__analysis_windows') }}

),

holdout_occasions as (

    select
        occasions.source,
        occasions.customer_id,
        occasions.gross_amount
    from {{ ref('int_customers__purchase_occasions') }} as occasions
    inner join windows
        on occasions.source = windows.source
    where occasions.order_date between windows.holdout_start and windows.holdout_end

),

aggregated as (

    select
        source,
        customer_id,
        count(*) as holdout_frequency,
        sum(gross_amount) as holdout_spend
    from holdout_occasions
    group by source, customer_id

)

select
    customers.source,
    customers.customer_id,
    coalesce(aggregated.holdout_frequency, 0) as holdout_frequency,
    cast(coalesce(aggregated.holdout_spend, 0) as decimal(12, 2)) as holdout_spend,

    -- Mean value per holdout occasion, for scoring the Gamma-Gamma monetary prediction. Zero when
    -- the customer did not buy, which Phase 4 must exclude rather than average in as a real zero.
    cast(
        case
            when coalesce(aggregated.holdout_frequency, 0) > 0
                then aggregated.holdout_spend / aggregated.holdout_frequency
            else 0
        end as decimal(12, 4)
    ) as holdout_monetary_value,

    windows.duration_holdout

from customers
inner join windows
    on customers.source = windows.source
left join aggregated
    on customers.source = aggregated.source
    and customers.customer_id = aggregated.customer_id

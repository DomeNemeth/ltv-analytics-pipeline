{{ config(tags=['post_fit']) }}

-- The predictions table holds what the last fit wrote, and nothing has edited it since.
--
-- assert_predictions_match_the_current_calibration_inputs guards the inputs. This guards the
-- output, which nothing else did. The Phase 4 audit multiplied every prediction by 1.167 on a
-- copy of the warehouse. The holdout error went from -14.3% to 0%, and every post-fit guard stayed
-- green, because nothing tied the predictions table to the fit that produced it.
--
-- `ltv fit` records the sum of expected purchases, a rank-weighted sum of the same, and the sum of
-- expected forward revenue, all across every horizon. Scaling moves the sums. Reassigning
-- predictions between customers moves the weighted sum. A left join, so a source with predictions
-- but no fit_runs row fails too.

with predictions as (

    select
        *,
        -- Dense, because each customer appears once per horizon and must carry the same weight
        -- in both. Matches PredictionFingerprint on the Python side.
        dense_rank() over (partition by source order by customer_id) as customer_rank
    from {{ ref('stg_model__customer_predictions') }}

),

written as (

    select
        source,
        sum(expected_purchases) as sum_expected_purchases,
        sum(customer_rank * expected_purchases) as weighted_expected_purchases,
        sum(expected_forward_revenue) as sum_expected_forward_revenue
    from predictions
    group by source

),

recorded as (

    select * from {{ source('model', 'fit_runs') }}

)

select
    written.source,
    recorded.sum_expected_purchases as recorded_purchases,
    written.sum_expected_purchases as table_purchases,
    recorded.sum_expected_forward_revenue as recorded_revenue,
    written.sum_expected_forward_revenue as table_revenue

from written

left join recorded
    on written.source = recorded.source

-- Relative tolerance for floating-point summation order, which differs between pandas and DuckDB.
-- The smallest edit worth catching moves a sum by far more than one part in a billion.
where recorded.source is null
    or abs(recorded.sum_expected_purchases - written.sum_expected_purchases)
        > 1e-9 * abs(recorded.sum_expected_purchases)
    or abs(recorded.weighted_expected_purchases - written.weighted_expected_purchases)
        > 1e-9 * abs(recorded.weighted_expected_purchases)
    or abs(recorded.sum_expected_forward_revenue - written.sum_expected_forward_revenue)
        > 1e-9 * abs(recorded.sum_expected_forward_revenue)

{{ config(tags=['post_fit']) }}

-- The Phase 3 predictions, brought into the dbt DAG.
--
-- Rename and cast only, no joins, per the staging convention. It exists so that nothing downstream
-- calls source() -- CLAUDE.md section 5 requires marts to reach the warehouse through the layers
-- rather than around them, and the scoring model and the Phase 5 marts both read predictions.
--
-- Columns are listed explicitly rather than `select *` so that a column added or renamed by
-- `ltv fit` shows up as a dbt failure here, at the boundary, instead of silently changing the shape
-- of everything downstream.

select
    source,
    customer_id,
    horizon_days,
    expected_purchases,
    probability_alive,
    expected_avg_value,
    expected_forward_revenue,
    forward_revenue_hdi_low,
    forward_revenue_hdi_high,
    spend_estimate_source,
    is_gamma_gamma_eligible,
    fit_method

from {{ source('model', 'customer_predictions') }}

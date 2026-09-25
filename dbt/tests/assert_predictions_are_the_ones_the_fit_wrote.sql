{{ config(tags=['post_fit']) }}

-- The predictions table holds what the last fit wrote, and nothing has edited it since.
--
-- assert_predictions_match_the_current_calibration_inputs guards the inputs. This guards the
-- output, which nothing else did. The Phase 4 audit multiplied every prediction by 1.167 on a
-- copy of the warehouse. The holdout error went from -14.3% to 0%, and every post-fit guard stayed
-- green, because nothing tied the predictions table to the fit that produced it.
--
-- The first version compared expected purchases and a plain revenue sum. The re-audit got three
-- more edits past it: swapping the 273- and 365-day labels, moving revenue between customers, and
-- overwriting expected_avg_value and probability_alive. So every column the report reads now gets
-- three sums: plain, weighted by customer rank, and weighted by horizon. See PredictionFingerprint
-- for what each one catches.
--
-- This column list must match FINGERPRINTED_PREDICTIONS in src/ltv/models/clv.py.
-- tests/test_fingerprint.py fails if they differ.
{% set columns = [
    'expected_purchases',
    'expected_forward_revenue',
    'expected_avg_value',
    'probability_alive',
] %}
{% set weightings = {
    'sum': '1',
    'weighted': 'customer_rank',
    'horizon_weighted': 'horizon_days',
} %}

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
        {%- for column in columns %}
        {%- for prefix, weight in weightings.items() %}
        sum({{ weight }} * cast({{ column }} as double)) as {{ prefix }}_{{ column }}
        {%- if not loop.last %},{% endif %}
        {%- endfor %}
        {%- if not loop.last %},{% endif %}
        {%- endfor %}
    from predictions
    group by source

),

recorded as (

    select * from {{ source('model', 'fit_runs') }}

)

select written.source

from written

left join recorded
    on written.source = recorded.source

-- A left join, so a source with predictions but no fit_runs row fails too. Relative tolerance for
-- floating-point summation order, which differs between pandas and DuckDB. The smallest edit worth
-- catching moves a sum by far more than one part in a billion. coalesce(..., true), so a NULL on
-- either side counts as a mismatch rather than as "no difference".
where recorded.source is null
{%- for column in columns %}
{%- for prefix in weightings %}
    or coalesce(
        abs(recorded.{{ prefix }}_{{ column }} - written.{{ prefix }}_{{ column }})
            > 1e-9 * abs(recorded.{{ prefix }}_{{ column }}),
        true
    )
{%- endfor %}
{%- endfor %}

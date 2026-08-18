-- The single definition of the calibration/holdout split, one row per source.
--
-- Every window-aware model cross-joins this instead of re-deriving dates. Two models each computing
-- "the cutoff" independently is how a calibration window and a holdout window end up overlapping by
-- a day, which leaks future purchases into the training features and inflates every metric in the
-- project with nothing failing.
--
-- The window is derived from the data rather than hardcoded, so it transfers to Online Retail in
-- Phase 7 unchanged. For CDNOW it happens to land exactly on the canonical Fader & Hardie split --
-- the file spans 78 weeks to the day, so 39 weeks of calibration ends on 1997-09-30 with no
-- rounding. assert_calibration_window_matches_benchmark pins that.

with observed as (

    select
        source,
        min(order_date) as first_order_date,
        max(order_date) as last_order_date
    from {{ ref('int_transactions__unioned') }}
    group by source

),

bounds as (

    select
        source,
        first_order_date,
        last_order_date,
        first_order_date as calibration_start,

        -- Inclusive of the first day: day 1 through day 273 is 39 whole weeks, so the offset is
        -- one less than the week count times seven. Off by one here silently shifts every
        -- customer's age and recency.
        --
        -- Cast back to date because DuckDB promotes date + interval to a timestamp, and a window
        -- boundary that is secretly a timestamp makes `between` comparisons against dates behave
        -- differently the moment a source carries a time component.
        cast(
            first_order_date + to_days({{ (var('calibration_weeks') | int) * 7 - 1 }}) as date
        ) as calibration_end,

        {{ (var('holdout_weeks') | int) * 7 }} as duration_holdout

    from observed

)

select
    source,
    calibration_start,
    calibration_end,
    cast(calibration_end + to_days(1) as date) as holdout_start,
    cast(calibration_end + to_days(duration_holdout) as date) as holdout_end,
    duration_holdout,

    -- The observed extent of the data, kept alongside the derived window so tests can assert the
    -- two agree rather than trusting the arithmetic.
    first_order_date,
    last_order_date

from bounds

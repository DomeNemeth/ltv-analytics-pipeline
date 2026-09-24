-- Every row of the window sensitivity, recomputed without the recent_repeats macro.
--
-- The report's crossover sentence ("beats the model only with a look-back of N days or less") is
-- computed from every row of int_baselines__recent_window_sensitivity. The Phase 4 audit's third
-- pass found that only the configured window's row was checked against anything. Scaling every
-- other row by 0.8 kept every guard green and moved the crossover. So each row is recomputed here
-- from the occasions and the analysis windows directly. This is written independently of the macro:
-- `date_diff(...) < window_days` rather than the macro's `order_date > calibration_end - window`, so
-- a defect in either one shows up as a disagreement.
--
-- The window list is read from the model's own rows rather than repeated here, so a window added
-- to the model is checked without touching this test.

with sensitivity as (

    select * from {{ ref('int_baselines__recent_window_sensitivity') }}

),

calibration as (

    select * from {{ ref('int_customers__rfm_calibration') }}

),

windows as (

    select * from {{ ref('int_sources__analysis_windows') }}

),

per_customer as (

    select
        sensitivity.source,
        sensitivity.window_days,
        least(sensitivity.window_days, calibration.customer_age) as exposure_days,
        (
            select count(*)
            from {{ ref('int_customers__purchase_occasions') }} as occasions
            where occasions.source = calibration.source
                and occasions.customer_id = calibration.customer_id
                and occasions.order_date > calibration.first_order_date
                and occasions.order_date <= windows.calibration_end
                and date_diff('day', occasions.order_date, windows.calibration_end)
                    < sensitivity.window_days
        ) as repeats,
        windows.duration_holdout

    from sensitivity

    inner join calibration
        on sensitivity.source = calibration.source

    inner join windows
        on sensitivity.source = windows.source

),

recomputed as (

    select
        source,
        window_days,
        sum(
            case
                when exposure_days > 0
                    then cast(repeats as double) / exposure_days * duration_holdout
                else 0.0
            end
        ) as predicted_purchases
    from per_customer
    group by source, window_days

)

select
    sensitivity.source,
    sensitivity.window_days,
    sensitivity.predicted_purchases,
    recomputed.predicted_purchases as expected_predicted_purchases

from sensitivity

left join recomputed
    on sensitivity.source = recomputed.source
    and sensitivity.window_days = recomputed.window_days

where coalesce(
    abs(sensitivity.predicted_purchases - recomputed.predicted_purchases)
        > 1e-9 * abs(recomputed.predicted_purchases),
    true
)

{#
    Repeat occasions per calibration customer in the last `window_days` days before the cutoff, and
    the exposure they were counted over.

    This feeds the recent-run-rate baseline: "a customer keeps repeating at the rate of their last
    quarter". It is the naive rule that exploits the non-stationarity Phase 4 diagnosed. The
    purchase rate falls steeply through calibration, so every rule averaged over the whole window
    overshoots, while a rule that looks only at the end of the window does not. The Phase 4 audit
    found it closer on the total than the model. So it is reported, not left out.

    One definition used twice, by int_customers__scored and by the window-sensitivity model. If
    the two drifted apart, the sensitivity sentence in the report would describe a different rule
    from the one scored in the tables above it.

    Exposure is the smaller of the window and the customer's age at the cutoff. A customer
    acquired inside the window cannot have repeated before their first purchase, so dividing by the
    whole window would understate their rate. Every CDNOW customer was acquired by 1997-03-25, so
    there it is always the full window. It will not be the full window for Online Retail.

    Calibration occasions only: bounded above by calibration_end. Holdout columns are not in scope.

    Args:
        window_days: length of the look-back window, in days, ending on calibration_end inclusive.
#}

{% macro recent_repeats(window_days) %}

    select
        calibration.source,
        calibration.customer_id,
        {{ window_days }} as window_days,
        count(occasions.order_date) as recent_repeats,
        least({{ window_days }}, calibration.customer_age) as recent_exposure_days

    from {{ ref('int_customers__rfm_calibration') }} as calibration

    inner join {{ ref('int_sources__analysis_windows') }} as windows
        on calibration.source = windows.source

    -- A left join, so a customer with no recent repeat keeps their row with a count of zero. An
    -- inner join would drop them, and the rule would then predict only for customers who were
    -- still buying.
    left join {{ ref('int_customers__purchase_occasions') }} as occasions
        on calibration.source = occasions.source
        and calibration.customer_id = occasions.customer_id
        -- Repeats only. The first occasion is excluded, as in calibration frequency.
        and occasions.order_date > calibration.first_order_date
        and occasions.order_date > windows.calibration_end - to_days({{ window_days }})
        and occasions.order_date <= windows.calibration_end

    group by calibration.source, calibration.customer_id, calibration.customer_age

{% endmacro %}

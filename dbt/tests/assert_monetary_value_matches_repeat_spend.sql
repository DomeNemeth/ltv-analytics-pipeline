-- Gamma-Gamma's monetary_value is the mean spend over REPEAT occasions: the first purchase is
-- excluded from the numerator and from the denominator. Both halves of that were unguarded.
--
-- assert_rfm_invariants_hold checks only the sign of monetary_value and the frequency = 0 case, so
-- dividing by `occasions` instead of `frequency` -- a one-word edit producing valid SQL and entirely
-- plausible numbers -- passed all 62 tests. It was caught only by the three hand-computed customers
-- in tests/test_rfm_math.py, and three customers out of 23,570 is a sample, not a guard. Phase 3
-- fits Gamma-Gamma on exactly this column, so an error here corrupts the model rather than the
-- report.
--
-- This recomputes both quantities from the occasion grain with separate arithmetic rather than
-- re-deriving them from the macro's own outputs, so an error in the macro cannot cancel itself out.
-- Scoped to the calibration window because that is the window the model is fitted on.

with windows as (

    select
        source,
        calibration_start,
        calibration_end
    from {{ ref('int_sources__analysis_windows') }}

),

calibration_occasions as (

    select
        occasions.source,
        occasions.customer_id,
        occasions.order_date,
        occasions.gross_amount

    from {{ ref('int_customers__purchase_occasions') }} as occasions
    inner join windows
        on occasions.source = windows.source
    where occasions.order_date between windows.calibration_start and windows.calibration_end

),

recomputed as (

    select
        source,
        customer_id,
        count(*) - 1 as repeat_occasions,
        sum(gross_amount) - arg_min(gross_amount, order_date) as repeat_spend

    from calibration_occasions
    group by source, customer_id

)

select
    rfm.source,
    rfm.customer_id,
    rfm.frequency,
    recomputed.repeat_occasions,
    rfm.monetary_value,
    recomputed.repeat_spend

from {{ ref('int_customers__rfm_calibration') }} as rfm

inner join recomputed
    on rfm.source = recomputed.source
    and rfm.customer_id = recomputed.customer_id

where
    -- The denominator is the repeat count, not the occasion count.
    rfm.frequency != recomputed.repeat_occasions

    -- And the numerator excludes the first purchase. Multiplying back out avoids dividing by zero
    -- for one-time buyers, whose monetary_value is 0 by definition and whose repeat_spend is 0 too.
    or abs(rfm.monetary_value * recomputed.repeat_occasions - recomputed.repeat_spend) > 0.01

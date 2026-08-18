-- Invariants that must hold for every customer in every RFM window, in both models.
--
-- These are the arithmetic identities BG/NBD depends on. Violating any of them means the window
-- bounds or the date arithmetic are wrong, and the resulting fit is meaningless -- but the numbers
-- still look like plausible numbers, so nothing else catches it.

with summaries as (

    select 'calibration' as window_name, * from {{ ref('int_customers__rfm_calibration') }}
    union all
    select 'full' as window_name, * from {{ ref('int_customers__rfm_full') }}

)

select
    window_name,
    source,
    customer_id,
    frequency,
    recency,
    customer_age,
    monetary_value

from summaries

where
    -- A customer cannot be older at their last purchase than they are at the window end.
    recency > customer_age

    -- Ages and repeat counts are counts of elapsed things. Negative means the window bounds are
    -- inverted or the first/last dates got swapped.
    or recency < 0
    or customer_age < 0
    or frequency < 0

    -- Frequency counts repeat occasions, so it is always one less than total occasions.
    or frequency != occasions - 1

    -- A one-time buyer has no repeat purchases and therefore no repeat spend to average. A non-zero
    -- monetary value here means the first purchase leaked into the Gamma-Gamma numerator.
    or (frequency = 0 and monetary_value != 0)

    -- Recency is only zero for one-time buyers; anyone with a repeat purchase bought on a later day
    -- than their first, because the grain is one row per customer per day.
    or (frequency > 0 and recency <= 0)

    -- Eligibility must agree with the quantities it is derived from.
    or (is_gamma_gamma_eligible and (frequency = 0 or monetary_value <= 0))

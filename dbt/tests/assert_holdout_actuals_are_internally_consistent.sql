-- holdout_monetary_value is what Phase 4 scores the Gamma-Gamma prediction against, and before this
-- test it had no coverage of any kind -- not a dbt test, not a Python test, not even a description
-- in the schema file. This asserts the three things that must hold for it to mean anything.
--
-- Note what is deliberately NOT asserted: that holdout_monetary_value is strictly positive whenever
-- the customer bought. CDNOW's 80 zero-value rows are all standalone occasions, so a customer whose
-- only holdout purchase was a $0.00 giveaway legitimately has frequency 1 and mean value 0. A test
-- demanding positivity would be asserting something false about the data.

with actuals as (

    select * from {{ ref('int_customers__holdout_actuals') }}

)

select
    source,
    customer_id,
    holdout_frequency,
    holdout_spend,
    holdout_monetary_value

from actuals

where
    -- A count and a sum of non-negative amounts cannot be negative. Returns would make spend
    -- negative, and CDNOW has none -- Phase 7's Online Retail does, and this test is where that
    -- decision gets forced into the open rather than quietly changing what the column means.
    holdout_frequency < 0
    or holdout_spend < 0

    -- A customer who did not buy in the holdout window must carry clean zeros, not a stray amount
    -- from a coalesce that lost its default.
    or (holdout_frequency = 0 and (holdout_spend != 0 or holdout_monetary_value != 0))

    -- The mean must reconstruct the total. This is what catches a wrong denominator, which is
    -- otherwise invisible: dividing by the wrong count produces a plausible number in a plausible
    -- range and every other test still passes. The cent of tolerance is for the 4-decimal rounding
    -- in the mean itself, nothing more.
    or abs(holdout_monetary_value * holdout_frequency - holdout_spend) > 0.01

-- Proves the line-item to purchase-occasion collapse both happened and was lossless.
--
-- Two failure modes, both silent without this test:
--
-- 1. The collapse stops happening (someone "simplifies" the GROUP BY, or a join fans out). BG/NBD
--    then counts line items as separate purchases, inflating frequency for the 1,774 CDNOW
--    customer-days that hold more than one row, and every model metric improves for the wrong
--    reason.
-- 2. The collapse loses rows. Summing line_items back up must reproduce the source row count
--    exactly -- if an inner join silently dropped transactions, spend and frequency both fall and
--    nothing else here would notice.
-- 3. The collapse loses money or units. Counting rows says nothing about what they carried:
--    replacing sum(gross_amount) with max(gross_amount) is valid SQL, produces entirely plausible
--    output, and reconciles perfectly on every count -- it passed all 62 tests before the amount
--    checks below were added. Only the three hand-computed customers in tests/test_rfm_math.py
--    noticed, and three out of 23,570 is a sample rather than a guard.

with occasions as (

    select
        count(*) as occasion_count,
        sum(line_items) as line_item_count,
        sum(quantity) as occasion_quantity,
        sum(gross_amount) as occasion_amount
    from {{ ref('int_customers__purchase_occasions') }}

),

transactions as (

    select
        count(*) as transaction_count,
        sum(quantity) as transaction_quantity,
        sum(gross_amount) as transaction_amount
    from {{ ref('int_transactions__unioned') }}

)

select
    occasions.occasion_count,
    occasions.line_item_count,
    transactions.transaction_count,
    occasions.occasion_quantity,
    transactions.transaction_quantity,
    occasions.occasion_amount,
    transactions.transaction_amount

from occasions
cross join transactions

where
    -- Rows were lost or duplicated on the way through.
    occasions.line_item_count != transactions.transaction_count

    -- Or nothing was collapsed at all, which on this data means the grain is wrong.
    or occasions.occasion_count >= occasions.line_item_count

    -- Or the rows survived but their contents did not. A collapse is only lossless if the money and
    -- the units come through it unchanged, which is the half of the claim this test's own header
    -- used to make without checking.
    or occasions.occasion_quantity != transactions.transaction_quantity
    or occasions.occasion_amount != transactions.transaction_amount

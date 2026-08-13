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

with occasions as (

    select
        count(*) as occasion_count,
        sum(line_items) as line_item_count
    from {{ ref('int_customers__purchase_occasions') }}

),

transactions as (

    select count(*) as transaction_count
    from {{ ref('int_transactions__unioned') }}

)

select
    occasions.occasion_count,
    occasions.line_item_count,
    transactions.transaction_count

from occasions
cross join transactions

where
    -- Rows were lost or duplicated on the way through.
    occasions.line_item_count != transactions.transaction_count

    -- Or nothing was collapsed at all, which on this data means the grain is wrong.
    or occasions.occasion_count >= occasions.line_item_count

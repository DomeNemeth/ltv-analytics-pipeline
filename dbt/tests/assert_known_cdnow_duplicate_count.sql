-- Pins a documented property of the CDNOW source file: exactly 255 rows are byte-identical
-- repeats of another row (215 groups, 470 rows, 255 of them excess).
--
-- These are kept rather than deduplicated. CDNOW records line items and carries no order
-- identifier, so two CDs at the same price on the same day is ordinary retail, not proof of a
-- double-load -- and dropping them would understate real spend by $4,332 across 164 customers.
--
-- The count is asserted because the decision to keep them only holds while the number is what we
-- think it is. If the upstream file changes, or the loader starts duplicating rows, this fails and
-- the decision gets revisited instead of silently continuing to apply to different data.

with grouped as (

    select count(*) as copies
    from {{ ref('stg_cdnow__transactions') }}
    group by customer_id, order_date, quantity, gross_amount
    having count(*) > 1

),

tally as (

    select coalesce(sum(copies - 1), 0) as excess_rows
    from grouped

)

select
    excess_rows,
    255 as expected_excess_rows
from tally
where excess_rows != 255

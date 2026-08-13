-- Line items collapsed to purchase occasions: one row per customer per day they bought anything.
--
-- This is the most consequential model in the project. BG/NBD counts purchase *occasions*, not line
-- items, and CDNOW has 1,774 customer-days holding more than one row. Feeding it line items would
-- inflate frequency for those customers by 2,068 events (3.0% of the file) and produce a model that
-- scores better than it deserves to, with every test still green.
--
-- Everything frequency-related reads this model. Nothing computing a customer metric may read
-- int_transactions__unioned directly.

with transactions as (

    select * from {{ ref('int_transactions__unioned') }}

),

occasions as (

    select
        source,
        customer_id,
        order_date,

        -- Returns are summed rather than filtered, so a refund nets against the day it lands on
        -- instead of vanishing. CDNOW contains none; Phase 7 revisits this, because Online Retail
        -- books cancellations on their own later invoice rather than against the original day.
        sum(quantity) as quantity,
        sum(gross_amount) as gross_amount,

        -- How many source rows collapsed into this occasion. Kept because it is what makes the
        -- collapse auditable: assert_occasions_collapsed reads it, and a reviewer can see at a
        -- glance that 3.0% of rows were absorbed rather than taking the claim on trust.
        count(*) as line_items

    from transactions
    group by source, customer_id, order_date

)

select
    -- One order per customer-day for CDNOW by construction, so this is also the order_id. It stops
    -- being so in Phase 7, where a customer can raise two invoices in a day and the occasion is the
    -- coarser grain of the two.
    source || '-' || cast(customer_id as varchar) || '-' || strftime(order_date, '%Y%m%d')
        as occasion_id,
    source,
    customer_id,
    order_date,
    quantity,
    cast(gross_amount as decimal(12, 2)) as gross_amount,
    line_items
from occasions

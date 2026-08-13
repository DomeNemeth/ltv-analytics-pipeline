-- The single cross-source union point. Every model downstream of here is source-agnostic and reads
-- this rather than any individual staging model.
--
-- It is a one-branch union today, which looks like indirection until Phase 7: adding Online Retail
-- means writing its staging model to the same contract and adding one `union all` here. Nothing
-- downstream changes. That is the whole return on the staging contract.

with cdnow as (

    select * from {{ ref('stg_cdnow__transactions') }}

)

select
    source,
    customer_id,
    order_id,
    order_date,
    quantity,
    unit_price,
    gross_amount,
    is_return
from cdnow

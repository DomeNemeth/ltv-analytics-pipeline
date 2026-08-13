-- CDNOW line items, conformed to the staging contract.
--
-- Rename, cast, clean. No joins, no aggregation, no filtering: the 255 byte-identical duplicate
-- rows and the 80 zero-value rows both survive this model intact and are handled downstream where
-- the decision is explicit. See CLAUDE.md section 5.

with source as (

    select * from {{ source('raw', 'cdnow_transactions') }}

),

renamed as (

    select
        'cdnow' as source,

        cast(customer_id as integer) as customer_id,

        -- CDNOW records line items and has no order identifier, so one is derived. This makes an
        -- "order" mean "everything this customer bought on this day", which is exactly the purchase
        -- occasion BG/NBD counts.
        --
        -- The contract column is real work, not a placeholder: Online Retail's InvoiceNo drops into
        -- this slot unchanged in Phase 7, and there a customer genuinely can raise two invoices in
        -- one day -- so the occasion collapse stays keyed on customer and date, never on order_id.
        cast(customer_id as varchar) || '-' || strftime(order_date, '%Y%m%d') as order_id,

        cast(order_date as date) as order_date,

        cast(quantity as integer) as quantity,

        -- An *average* unit price: the source gives a line total for `quantity` CDs, which need not
        -- all have been the same price. Four decimal places rather than two because this is a
        -- derived rate, and rounding it to cents would imply a precision it does not have.
        --
        -- nullif guards a division that cannot fail on this dataset (quantity is always positive).
        -- If that ever changes, the not_null test on this column fails loudly, which is the point.
        cast(gross_amount / nullif(quantity, 0) as decimal(10, 4)) as unit_price,

        cast(gross_amount as decimal(10, 2)) as gross_amount,

        -- CDNOW contains no returns or cancellations: every quantity is positive and every amount
        -- is non-negative. Online Retail encodes them as negative quantities in Phase 7.
        false as is_return

    from source

)

select * from renamed

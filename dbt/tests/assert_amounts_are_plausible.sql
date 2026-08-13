-- Sign and magnitude checks on money and quantities.
--
-- Written to survive Phase 7: Online Retail encodes cancellations as negative quantities, so the
-- non-negativity rule applies only to rows not flagged as returns. Returns get their own checks
-- when that source lands.

select
    source,
    order_id,
    customer_id,
    order_date,
    quantity,
    unit_price,
    gross_amount,
    is_return

from {{ ref('int_transactions__unioned') }}

where
    not is_return
    and (
        gross_amount < 0
        or quantity <= 0

        -- A unit price cannot be negative when neither of its inputs is, and it is null only when
        -- quantity was zero -- which the check above already rejects, so a null here means the
        -- guard in staging was removed.
        or unit_price < 0
        or unit_price is null
    )

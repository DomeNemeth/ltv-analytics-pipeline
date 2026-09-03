{{ config(tags=['post_fit']) }}

-- The forecast being scored must cover exactly the window it is scored against.
--
-- `ltv fit` writes predictions at two horizons: the holdout length and the headline LTV horizon
-- (273 and 365 days on CDNOW). Both are valid, both are plausible, and scoring the 365-day forecast
-- against the 273-day holdout would make the model look 34% worse than it is with nothing to
-- indicate why. The reverse mistake in a later phase would flatter it instead.
--
-- `is distinct from` rather than `!=` on purpose: a NULL horizon_days means the predictions join
-- found nothing at all, which must fail here rather than pass on a comparison that is neither true
-- nor false.

select
    scored.source,
    scored.customer_id,
    scored.horizon_days,
    windows.duration_holdout

from {{ ref('int_customers__scored') }} as scored

inner join {{ ref('int_sources__analysis_windows') }} as windows
    on scored.source = windows.source

where scored.horizon_days is distinct from windows.duration_holdout

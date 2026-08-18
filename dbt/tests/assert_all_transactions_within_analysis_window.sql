-- Every transaction must fall inside the analysis window, i.e. calibration plus holdout together.
--
-- This is a configuration check, not a data check. The window is derived from the first order date
-- plus calibration_weeks + holdout_weeks, so if those are set too small for a source, the tail of
-- the data falls outside both windows and is silently ignored by every downstream model. Row counts
-- stay plausible, the build stays green, and the model is trained on a truncated dataset.
--
-- For CDNOW the 39/39 split covers the file exactly, so there is no slack here at all: a one-week
-- misconfiguration fails this test.

select
    transactions.source,
    min(transactions.order_date) as earliest_order,
    max(transactions.order_date) as latest_order,
    windows.calibration_start,
    windows.holdout_end

from {{ ref('int_transactions__unioned') }} as transactions

inner join {{ ref('int_sources__analysis_windows') }} as windows
    on transactions.source = windows.source

group by transactions.source, windows.calibration_start, windows.holdout_end

having
    min(transactions.order_date) < windows.calibration_start
    or max(transactions.order_date) > windows.holdout_end

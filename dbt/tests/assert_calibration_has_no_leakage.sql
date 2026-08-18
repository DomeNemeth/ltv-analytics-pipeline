-- The leakage guard, and the most important test in the project.
--
-- No feature computed for the calibration window may be influenced by anything that happened after
-- the cutoff. Leakage does not break a build or produce an obviously wrong number: it produces a
-- model that scores beautifully on holdout and is worthless in production, because it was quietly
-- told the answer. Nothing else here would catch it.
--
-- Fails if any customer's last calibration order lands after calibration_end.

select
    rfm.source,
    max(rfm.last_order_date) as latest_calibration_order,
    windows.calibration_end

from {{ ref('int_customers__rfm_calibration') }} as rfm

inner join {{ ref('int_sources__analysis_windows') }} as windows
    on rfm.source = windows.source

group by rfm.source, windows.calibration_end
having max(rfm.last_order_date) > windows.calibration_end

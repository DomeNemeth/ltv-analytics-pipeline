-- Pins the CDNOW analysis window to the canonical Fader & Hardie split.
--
-- The window is derived from the data rather than hardcoded, which is what makes it transfer to a
-- second source. This test is the other half of that trade: it proves the derivation still lands on
-- the published benchmark split, so the Phase 4 error metrics stay comparable to the literature.
--
-- The CDNOW master file spans 1997-01-01 to 1998-06-30, which is 78 weeks to the day, so 39 weeks
-- of calibration ends on 1997-09-30 exactly with no rounding.
--
-- Scoped to CDNOW: other sources have their own windows and no benchmark to match.

select
    source,
    calibration_start,
    calibration_end,
    holdout_start,
    holdout_end

from {{ ref('int_sources__analysis_windows') }}

where source = 'cdnow'
    and (
        calibration_start != date '1997-01-01'
        or calibration_end != date '1997-09-30'
        or holdout_start != date '1997-10-01'
        or holdout_end != date '1998-06-30'
    )

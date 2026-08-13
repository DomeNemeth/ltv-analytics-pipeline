{{ config(materialized='table') }}

-- BG/NBD and Gamma-Gamma inputs, computed over the calibration window only.
--
-- Read by the Phase 3 model fit, so it is materialized as a table rather than recomputed on every
-- query from Python.
--
-- Nothing in this model may reference a date after calibration_end. The window bounds come from
-- int_sources__analysis_windows and are applied inside the macro; assert_calibration_has_no_leakage
-- proves it held.

{{ rfm_summary('calibration_start', 'calibration_end') }}

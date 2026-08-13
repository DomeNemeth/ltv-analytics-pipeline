{{ config(materialized='table') }}

-- The same RFM summary over the entire observed period, calibration and holdout together.
--
-- This is deliberately NOT a model input. It exists so Phase 4 can compare what the model predicted
-- from the calibration window against what the customer actually went on to do, and so the marts
-- can describe a customer's real lifetime rather than their first 39 weeks.
--
-- Using this to fit anything would be leakage of the most direct kind.

{{ rfm_summary('calibration_start', 'holdout_end') }}

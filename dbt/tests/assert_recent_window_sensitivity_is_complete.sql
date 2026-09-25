-- One row per (source, window_days), each totalled over the whole calibration population.
--
-- The second condition catches what a grain test cannot. recent_repeats() left-joins occasions
-- onto the calibration population so that customers with no recent repeat keep a zero. Turn that
-- into an inner join and the rows stay unique while the population shrinks to recent buyers only.
-- The predicted total barely moves, because the dropped customers contribute zero. Only the
-- customer count shows it.

with sensitivity as (

    select * from {{ ref('int_baselines__recent_window_sensitivity') }}

),

population as (

    select source, count(*) as customers
    from {{ ref('int_customers__rfm_calibration') }}
    group by source

),

duplicated as (

    select source, window_days
    from sensitivity
    group by source, window_days
    having count(*) > 1

)

select sensitivity.source, sensitivity.window_days, 'duplicate row' as problem
from sensitivity
inner join duplicated
    on sensitivity.source = duplicated.source
    and sensitivity.window_days = duplicated.window_days

union all

select sensitivity.source, sensitivity.window_days, 'population incomplete' as problem
from sensitivity
left join population
    on sensitivity.source = population.source
where population.customers is null
    or sensitivity.customers != population.customers

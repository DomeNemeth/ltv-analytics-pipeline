{#
    Per-customer RFM summary over an arbitrary window.

    Both int_customers__rfm_calibration and int_customers__rfm_full are this macro with different
    bounds. They must not be written out twice: Phase 4 compares calibration features against
    full-period actuals, and if the two definitions drifted apart that comparison would silently
    stop measuring anything.

    Args:
        period_start_column: column of int_sources__analysis_windows opening the window.
        period_end_column:   column of int_sources__analysis_windows closing the window, inclusive.
#}

{% macro rfm_summary(period_start_column, period_end_column) %}

with occasions as (

    select * from {{ ref('int_customers__purchase_occasions') }}

),

windows as (

    select
        source,
        {{ period_start_column }} as period_start,
        {{ period_end_column }} as period_end
    from {{ ref('int_sources__analysis_windows') }}

),

in_period as (

    select
        occasions.source,
        occasions.customer_id,
        occasions.order_date,
        occasions.gross_amount,
        windows.period_end
    from occasions
    inner join windows
        on occasions.source = windows.source
    where occasions.order_date between windows.period_start and windows.period_end

),

summarised as (

    select
        source,
        customer_id,
        period_end,
        count(*) as occasions,

        -- BG/NBD models *repeat* purchasing, so frequency excludes the first occasion. A customer
        -- who bought once has frequency 0, not 1. This is the single most commonly mis-stated
        -- quantity in CLV work and it biases every parameter if it is wrong.
        count(*) - 1 as frequency,

        min(order_date) as first_order_date,
        max(order_date) as last_order_date,
        sum(gross_amount) as total_spend,

        -- Spend on the earliest occasion. The grain of the upstream model is one row per customer
        -- per day, so the argument to arg_min is unambiguous.
        arg_min(gross_amount, order_date) as first_occasion_spend

    from in_period
    group by source, customer_id, period_end

)

select
    source,
    customer_id,
    occasions,
    frequency,

    -- Recency in the BG/NBD sense: the customer's age at their last purchase, i.e. time between
    -- first and last occasion. Zero for one-time buyers. This is NOT days-since-last-purchase,
    -- which is what "recency" means in marketing RFM -- the two run in opposite directions.
    date_diff('day', first_order_date, last_order_date) as recency,

    -- "T" in Fader & Hardie notation: the customer's age at the end of the window. Named in full
    -- here because DuckDB folds unquoted identifiers to lowercase, and a column that has to be
    -- quoted as "T" everywhere is worse than one that says what it means.
    date_diff('day', first_order_date, period_end) as customer_age,

    first_order_date,
    last_order_date,
    cast(total_spend as decimal(12, 2)) as total_spend,

    -- Gamma-Gamma is fitted on the mean value of *repeat* transactions, so the first purchase is
    -- excluded from the numerator as well as the denominator. Zero for one-time buyers, who carry
    -- no information about repeat spend and are excluded from the fit below.
    cast(
        case
            when frequency > 0 then (total_spend - first_occasion_spend) / frequency
            else 0
        end as decimal(12, 4)
    ) as monetary_value,

    -- Gamma-Gamma requires at least one repeat purchase and a strictly positive mean value. On
    -- CDNOW this excludes the ~60% of customers who bought once, and the 68 customers whose entire
    -- history is promotional giveaways totalling $0.00.
    --
    -- Flagged rather than filtered on purpose: these customers still have a valid BG/NBD frequency,
    -- they still belong in customer counts, and a filter here would hide them from every downstream
    -- consumer. Phase 3 excludes them at fit time and reports how many it dropped.
    (frequency > 0 and (total_spend - first_occasion_spend) > 0) as is_gamma_gamma_eligible

from summarised

{% endmacro %}

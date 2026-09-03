"""Prove the test suite would catch the bugs it claims to catch.

A test that cannot fail is worse than no test, because it buys false confidence. Reading a test does
not tell you whether it can fail -- the Phase 1 review found three that could not, and reading had
missed all three. Deleting the logic and re-running does tell you.

Each mutation below breaks one specific piece of modelling logic in a way that leaves valid,
plausible-looking SQL. The suite must go red. If it stays green, the guard named alongside the
mutation is decorative and needs rewriting.

Run it directly; it restores every file it touches, including on interrupt:

    uv run python scripts/mutation_check.py
    uv run python scripts/mutation_check.py --only leakage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect, and the guard that is supposed to notice."""

    name: str
    path: str
    old: str
    new: str
    guard: str
    #: Override the default checks. Only for defects that cannot show up without a more expensive
    #: command -- a wrong fit-provenance record, for instance, does nothing until a fit writes one.
    checks: tuple[str, ...] | None = None

    @property
    def file(self) -> Path:
        return REPO_ROOT / self.path


MUTATIONS = (
    Mutation(
        name="collapse-grain",
        path="dbt/models/intermediate/int_customers__purchase_occasions.sql",
        old="    group by source, customer_id, order_date",
        new="    group by source, customer_id, order_date, quantity, gross_amount",
        # Verified: caught by the unique test on occasion_id (1,614 duplicates), not by
        # assert_occasions_collapsed -- that one still passes here, because summing line_items
        # continues to reconcile and the occasion count still falls below the line-item count.
        # Hence the separate mutation below, which is what actually exercises it.
        guard="unique occasion_id, and RFM math via inflated frequency",
    ),
    Mutation(
        name="collapse-lossless",
        path="dbt/models/intermediate/int_customers__purchase_occasions.sql",
        old="        count(*) as line_items",
        new="        1 as line_items",
        guard="assert_occasions_collapsed",
    ),
    Mutation(
        name="leakage",
        path="dbt/models/intermediate/int_customers__rfm_calibration.sql",
        old="{{ rfm_summary('calibration_start', 'calibration_end') }}",
        new="{{ rfm_summary('calibration_start', 'holdout_end') }}",
        guard="assert_calibration_has_no_leakage",
    ),
    Mutation(
        name="monetary",
        path="dbt/macros/rfm_summary.sql",
        old="when frequency > 0 then (total_spend - first_occasion_spend) / frequency",
        new="when frequency > 0 then total_spend / frequency",
        guard="RFM math (first purchase leaks into the Gamma-Gamma mean)",
    ),
    Mutation(
        name="window",
        path="dbt/models/intermediate/int_sources__analysis_windows.sql",
        old="to_days({{ (var('calibration_weeks') | int) * 7 - 1 }})",
        new="to_days({{ (var('calibration_weeks') | int) * 7 }})",
        guard="assert_calibration_window_matches_benchmark",
    ),
    Mutation(
        name="connection",
        path="src/ltv/transform.py",
        old="            _release_warehouse()",
        new="            pass",
        guard="RFM math fixture (dbt keeps the warehouse handle open)",
    ),
    # The four below were added after a whole-layer review, and every one of them was confirmed to
    # survive the entire suite beforehand. They are not hypothetical: each is a one-word edit that
    # produces valid SQL and plausible output, which is precisely the class of defect that reading
    # does not catch.
    Mutation(
        name="holdout-population",
        path="dbt/models/intermediate/int_customers__holdout_actuals.sql",
        old="left join aggregated",
        new="inner join aggregated",
        # The worst of the four. This silently drops the 16,512 customers who never came back --
        # 70% of the population, and specifically the ones the model is least able to predict, so
        # every holdout metric improves sharply. It passed all 62 tests until
        # assert_customer_populations_align existed.
        guard="assert_customer_populations_align",
    ),
    Mutation(
        name="holdout-spend",
        path="dbt/models/intermediate/int_customers__holdout_actuals.sql",
        old="        sum(gross_amount) as holdout_spend",
        new="        sum(gross_amount) / 2 as holdout_spend",
        guard="assert_spend_reconciles_across_windows",
    ),
    Mutation(
        name="collapse-amount",
        path="dbt/models/intermediate/int_customers__purchase_occasions.sql",
        old="        sum(gross_amount) as gross_amount,",
        new="        max(gross_amount) as gross_amount,",
        # Counts reconcile perfectly under this mutation; only the money is wrong. It is the
        # complement of collapse-lossless, which proves rows survive but says nothing about what
        # they carried.
        guard="assert_occasions_collapsed (amount reconciliation)",
    ),
    Mutation(
        name="monetary-denominator",
        path="dbt/macros/rfm_summary.sql",
        old="(total_spend - first_occasion_spend) / frequency",
        new="(total_spend - first_occasion_spend) / occasions",
        # The companion to `monetary` above, which mutates the numerator. Phase 3 fits Gamma-Gamma
        # on this column, so a wrong denominator corrupts the model rather than a report.
        guard="assert_monetary_value_matches_repeat_spend",
    ),
    # Phase 3. These live in Python rather than SQL, and the dbt suite cannot see any of them --
    # every one produces a model that fits, converges, and reports plausible numbers.
    Mutation(
        name="fit-leakage",
        path="src/ltv/models/clv.py",
        old='CALIBRATION_RELATION = "int_customers__rfm_calibration"',
        new='CALIBRATION_RELATION = "int_customers__rfm_full"',
        # The most damaging single edit available in this project: it trains on the holdout window,
        # so Phase 4's metrics come out excellent and mean nothing. Both relations are valid models
        # with identical schemas, so nothing downstream objects.
        guard="test_the_fit_reads_the_calibration_window_not_the_full_period",
    ),
    Mutation(
        name="rfm-rename",
        path="src/ltv/models/clv.py",
        old='BG_NBD_RENAMES = {"customer_age": "T"}',
        new='BG_NBD_RENAMES = {"recency": "T", "customer_age": "recency"}',
        # recency and T are both day counts over the same range, so swapping them fits cleanly and
        # predicts nonsense.
        guard="test_customer_age_becomes_T_and_recency_is_left_alone",
    ),
    Mutation(
        name="gamma-gamma-eligibility",
        path="src/ltv/models/clv.py",
        old="return self.eligible[list(GAMMA_GAMMA_COLUMNS)].reset_index(drop=True)",
        new="return self.customers[list(GAMMA_GAMMA_COLUMNS)].reset_index(drop=True)",
        # Fits the spend model on 14,120 customers whose repeat spend is zero, dragging the
        # population estimate down without erroring.
        guard="test_spend_model_sees_only_customers_with_repeat_spend",
    ),
    Mutation(
        name="population-fallback",
        path="src/ltv/models/clv.py",
        old="return aligned.fillna(population), fitted_individually",
        new="return aligned, fitted_individually",
        # Silently NULLs expected value for 60% of customers, so every segment total in the Phase 5
        # dashboard would cover 40% of the customer base while looking complete.
        guard="test_customers_without_repeat_spend_get_the_population_estimate",
    ),
    # Phase 4. These break the scoring rather than the modelling, which is the more dangerous half:
    # a wrong model produces numbers someone might question, while wrong scoring produces numbers
    # that make a wrong model look right. Note that `ltv transform` cannot see any of the SQL ones
    # -- it excludes tag:post_fit -- so `ltv validate` is what has to catch them.
    Mutation(
        name="scored-population",
        path="dbt/models/intermediate/int_customers__scored.sql",
        old="    select * from {{ ref('int_customers__rfm_calibration') }}",
        new=(
            "    select * from {{ ref('int_customers__rfm_calibration') }}"
            " where is_gamma_gamma_eligible"
        ),
        # Scores only the customers the spend model could be fitted on, dropping 14,120 of 23,570.
        # A plausible edit -- "score the customers we can actually model" -- and a catastrophic one:
        # the excluded group is 60% of the base and the part the model handles by population
        # fallback, so every metric in the report improves at once.
        #
        # Note what this mutation replaced. The obvious candidate was turning the `actuals` join
        # from `left` to `inner`, mirroring the Phase 2 defect. It was tried first and **survived**,
        # for a good reason: assert_customer_populations_align already pins the holdout and
        # calibration populations to the same set of keys, so at that join the two forms are
        # equivalent and nothing downstream can tell them apart. The `left join` there is
        # defensive, not load-bearing, and a mutation of it proves nothing about this guard.
        guard="assert_scored_population_is_complete",
    ),
    Mutation(
        name="scored-horizon",
        path="dbt/models/intermediate/int_customers__scored.sql",
        old="and predictions.horizon_days = windows.duration_holdout",
        new="and predictions.horizon_days = 365",
        # Scores the 365-day forecast against the 273-day outcome. Both horizons are real rows in
        # the predictions table, so the join succeeds and every customer still gets exactly one
        # prediction; the forecast is simply for the wrong window.
        guard="assert_scored_horizon_matches_the_holdout_window",
    ),
    Mutation(
        name="baseline-scale",
        path="dbt/models/intermediate/int_customers__scored.sql",
        # Single-line anchors throughout this file, deliberately. A multi-line `old` compares
        # against bytes read with newline="" -- so it stops matching the moment the file is written
        # with CRLF, which is exactly what happened here and reported the mutation as stale rather
        # than as surviving. A one-line anchor cannot have that problem.
        old="                * windows.duration_holdout",
        new="                * 1.0",
        # Drops the scaling to the holdout length, so the baseline predicts a per-day rate against a
        # 273-day actual and comes out ~273x too low. The model would then "beat the baseline" by a
        # margin nobody would question, which is exactly the wrong reason to believe a result.
        guard="assert_baselines_use_only_calibration_inputs",
    ),
    Mutation(
        name="carry-forward-baseline",
        path="dbt/models/intermediate/int_customers__scored.sql",
        old="    cast(calibration.frequency as double) as baseline_carry_forward,",
        new="    cast(calibration.frequency as double) * 1.5 as baseline_carry_forward,",
        # The fairest baseline in the report, and therefore the one whose corruption would flatter
        # the model most. A validation audit found that the two rate baselines carried a hidden
        # 273/229 inflation, which was worth roughly half the reported margin; this baseline exists
        # to be the un-inflated comparand, so it needs a guard proving nothing can inflate it.
        guard="assert_baselines_use_only_calibration_inputs",
    ),
    Mutation(
        name="zero-baseline",
        path="dbt/models/intermediate/int_customers__scored.sql",
        old="    0.0 as baseline_zero",
        new="    999.0 as baseline_zero",
        # The MAE floor. If this is wrong, the report's "the model clears the all-zero rule by 2%"
        # sentence becomes arbitrary, and per-customer MAE goes back to looking like accuracy.
        guard="assert_baselines_use_only_calibration_inputs",
    ),
    Mutation(
        name="mape-zeros",
        path="src/ltv/models/metrics.py",
        old="    defined = actual_values != 0",
        new="    defined = actual_values == actual_values",
        # Includes the customers whose actual is zero, making every percentage error infinite -- or,
        # with a different formulation, quietly enormous. The failure mode this guards is not the
        # NaN; it is a MAPE reported for a population it does not describe.
        guard="test_mape_excludes_zero_actuals_and_says_how_many",
    ),
    Mutation(
        name="stale-fit",
        path="src/ltv/models/store.py",
        old='"calibration_rows": [fingerprint.rows],',
        new='"calibration_rows": [0],',
        # The freshness guard's own guard. A fit that misrecords what it trained on cannot detect
        # that the warehouse moved underneath it, and the whole point of fit_runs is to make that
        # detectable. Needs `ltv fit` to run: the row is only written at fit time, so a validate
        # against the previous run's honest row would pass and the mutation would look survivable.
        checks=("ltv transform", "ltv fit", "ltv validate", "pytest"),
        guard="assert_predictions_match_the_current_calibration_inputs",
    ),
)

#: Every check the harness can run, in the order a real user would.
#:
#: `ltv validate` earns its place because `ltv transform` deliberately excludes `tag:post_fit`, so
#: the scoring model and its four tests are invisible to it. Without this entry, a mutation to
#: int_customers__scored would survive the whole harness while looking thoroughly checked.
#:
#: `ltv fit` is not run by default -- it is 30 seconds against everything else's five -- but a
#: mutation can opt into it by name when the defect only materialises at fit time.
CHECKS = {
    "ltv transform": ("uv", "run", "ltv", "transform"),
    "ltv fit": ("uv", "run", "ltv", "fit"),
    "ltv validate": ("uv", "run", "ltv", "validate"),
    "pytest": ("uv", "run", "pytest", "-q"),
}

DEFAULT_CHECKS = ("ltv transform", "ltv validate", "pytest")

#: What regenerates the *committed* reports and charts. Deliberately not the same command as the
#: `ltv validate` check above: the committed report has a model-comparison section in it, so a plain
#: validate rebuilds the numbers correctly and still leaves a report missing a section. Run once
#: at the end of a run rather than after every mutation -- per-mutation rebuilds only need to
#: remove the poison, and paying 35 seconds of challenger fitting nineteen times to do it would add
#: ten minutes for no extra safety.
FINAL_REBUILD = ("uv", "run", "ltv", "validate", "--compare-models")


def run_checks(names: Sequence[str]) -> list[str]:
    """Run the named checks in order, returning the names of those that failed.

    Every check runs even after one fails, because which of them notices is the interesting part.
    A mutation caught only by `pytest` and not by `ltv transform` says something specific about
    where the guard lives.
    """
    failed = []
    for name in names:
        completed = subprocess.run(  # noqa: S603
            CHECKS[name], cwd=REPO_ROOT, capture_output=True, text=True
        )
        if completed.returncode != 0:
            failed.append(name)
    return failed


def _rebuild_state(mutation: Mutation) -> None:
    """Rebuild everything downstream of the restored source, so no mutation outlives its own run.

    Restoring the *file* is not enough, and assuming it was cost this project a corrupted fit.
    A mutation that changes SQL leaves the warehouse **materialised from the mutated model** once
    the run finishes -- the source file is clean, `git diff` is clean, and the data is wrong. The
    last mutation of a run is the one that sticks, so the poison is whatever ran last.

    That is exactly what happened: a run ending on `monetary-denominator` left
    int_customers__rfm_calibration holding monetary_value divided by occasions instead of
    frequency. The next `ltv fit` read it, fitted Gamma-Gamma on values a third too low, and wrote a
    predictions table that looked entirely normal. Nothing in the repo could have noticed, because
    every check here inspects source rather than state.

    Phase 4 widened the blast radius, so this now rebuilds three kinds of state rather than one:

    * the **warehouse**, as before;
    * the **fitted models and their provenance row**, but only for a mutation that ran `ltv fit`
      -- a poisoned fit_runs row would otherwise fail the freshness guard on every later mutation
      and report a cascade of false catches;
    * the **committed reports and charts**, which `ltv validate` writes. A mutated run leaves
      reports/ holding numbers computed from broken code, and those files are tracked. Regenerating
      them here is the difference between a clean `git status` and a plausible-looking wrong report
      sitting in the working tree waiting to be committed.
    """
    steps = ["ltv transform"]
    if "ltv fit" in (mutation.checks or ()):
        steps.append("ltv fit")
    steps.append("ltv validate")

    for name in steps:
        subprocess.run(  # noqa: S603
            CHECKS[name], cwd=REPO_ROOT, capture_output=True, text=True
        )


def check_mutation(mutation: Mutation) -> bool:
    """Apply one mutation, run the suite, restore. True if the suite noticed."""
    # newline="" both ways, so line endings survive the round trip byte for byte. Without it, text
    # mode rewrites every line on Windows and the harness leaves half the dbt layer showing as
    # modified with an empty diff -- noise that makes a real uncommitted change easy to miss.
    # Path.read_text gained a newline argument only in 3.13, and this project pins 3.12.
    with mutation.file.open("r", encoding="utf-8", newline="") as handle:
        original = handle.read()
    if mutation.old not in original:
        raise SystemExit(
            f"{mutation.name}: the text to mutate is no longer in {mutation.path}. "
            f"The mutation is stale -- update it to match the current code.\n  {mutation.old!r}"
        )

    mutated = original.replace(mutation.old, mutation.new)
    try:
        mutation.file.write_text(mutated, encoding="utf-8", newline="")
        failed = run_checks(mutation.checks or DEFAULT_CHECKS)
    finally:
        mutation.file.write_text(original, encoding="utf-8", newline="")
        _rebuild_state(mutation)

    if failed:
        print(f"  caught by: {', '.join(failed)}")
    else:
        print(f"  NOT CAUGHT -- expected {mutation.guard} to fail")
    return bool(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run a single mutation by name")
    arguments = parser.parse_args()

    selected = [m for m in MUTATIONS if arguments.only in (None, m.name)]
    if not selected:
        raise SystemExit(f"no mutation named {arguments.only!r}")

    # Restore anything a previous interrupted run left behind before starting.
    subprocess.run(("git", "diff", "--quiet"), cwd=REPO_ROOT, check=False)

    survivors = []
    try:
        for mutation in selected:
            print(f"\n{mutation.name}: {mutation.path}")
            print(f"  expecting {mutation.guard}")
            if not check_mutation(mutation):
                survivors.append(mutation)
    finally:
        # Leave the working tree holding the reports a normal run produces, not the reduced ones
        # the per-mutation rebuild writes. Without this the harness ends with a committed report
        # whose model-comparison section reads "Not run" -- accurate for the command that wrote it,
        # wrong for the repo, and a diff that looks like a deliberate change. In the `finally` for
        # the same reason everything else here is: an interrupted run is when a half-restored tree
        # is most likely to be left behind and least likely to be noticed.
        print("\nregenerating committed reports ...")
        subprocess.run(FINAL_REBUILD, cwd=REPO_ROOT, capture_output=True, text=True)  # noqa: S603

    print(f"\n{len(selected) - len(survivors)}/{len(selected)} mutations caught")
    for mutation in survivors:
        print(f"  SURVIVED: {mutation.name} -- {mutation.guard} does not actually guard it")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())

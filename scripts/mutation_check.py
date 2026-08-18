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
)

CHECKS = (
    ("ltv transform", ("uv", "run", "ltv", "transform")),
    ("pytest", ("uv", "run", "pytest", "-q")),
)


def run_checks() -> list[str]:
    """Return the names of the checks that failed."""
    failed = []
    for name, command in CHECKS:
        completed = subprocess.run(  # noqa: S603
            command, cwd=REPO_ROOT, capture_output=True, text=True
        )
        if completed.returncode != 0:
            failed.append(name)
    return failed


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
        failed = run_checks()
    finally:
        mutation.file.write_text(original, encoding="utf-8", newline="")

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
    for mutation in selected:
        print(f"\n{mutation.name}: {mutation.path}")
        print(f"  expecting {mutation.guard}")
        if not check_mutation(mutation):
            survivors.append(mutation)

    print(f"\n{len(selected) - len(survivors)}/{len(selected)} mutations caught")
    for mutation in survivors:
        print(f"  SURVIVED: {mutation.name} -- {mutation.guard} does not actually guard it")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())

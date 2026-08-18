# CLAUDE.md — ltv-analytics-pipeline

## 1. What this is

An end-to-end pipeline that turns raw transaction data into per-customer lifetime-value predictions
and a published segment dashboard. It answers one business question: **which customer segments are
worth acquiring and retaining, and what is each segment's expected forward revenue?**

The point of difference is not that it fits a CLV model — it is that it **validates** one. Predictions
are made on a calibration window and scored against a held-out future window, with reported error
metrics and explicit assumption checks. Fitting is not predicting.

## 2. Stack, and why

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Warehouse | DuckDB | Zero infra, single file, Evidence reads it natively. BigQuery would add cloud setup and cost for a dataset that fits in memory. |
| Transformation | dbt-duckdb | Layered SQL with tests and lineage. |
| CLV model | PyMC-Marketing | `lifetimes` is unmaintained; PyMC-Marketing is its designated successor and gives posterior uncertainty, not just point estimates. |
| Sampling | nutpie + numba | Prebuilt wheels, so the project builds and samples **without a C compiler**. See §4. |
| Orchestration | Prefect | Earns its place for exactly one reason: the pipeline is `dbt → Python → dbt` (RFM features feed the model fit, whose predictions feed the marts). That handoff is awkward in pure dbt. |
| Dashboard | Evidence.dev | Builds a static site, so it publishes free to GitHub Pages. Streamlit would need a running server. |
| Packaging | uv | Fast, lockfile-based, pins the interpreter. |

**Prefect is capped at one flow file.** No deployments, no work pools, no agents, no Prefect Cloud.
If it grows past `flows/ltv_pipeline.py`, it has become a framework showcase and should be cut back.

**Ingestion is deliberately thin.** No Parquet archive layer, no incremental loading, no source
freshness scheduling, no connector abstraction. That ground is covered by a previous project; the
budget here goes to modelling and validation.

## 3. Commands

Everything runs through `uv`. There is no Makefile — commands must work identically on Windows,
macOS, and Linux.

```bash
uv sync                       # install the locked environment
uv run ltv info               # show resolved config and whether the warehouse is built
uv run ltv ingest cdnow       # download (checksum-pinned), parse, load raw.cdnow_transactions
uv run ltv transform          # dbt build: all models plus every dbt test
uv run pytest                 # Python tests; integration tests skip if source data is absent
uv run ruff check . && uv run ruff format --check .

uv run python scripts/mutation_check.py    # prove the guards can actually fail
```

Commands are added to the CLI as each stage lands, so `uv run ltv --help` always reflects what the
repo can actually do. It never advertises stubs.

**`ltv transform` is the only supported way to run dbt.** It passes anything after it straight
through, so `uv run ltv transform --select staging` and `uv run ltv transform test` both work. It
exists rather than a shell wrapper because it must behave identically on all three platforms and
because Phase 6's Prefect flow needs a Python entry point regardless. Calling `dbt` directly needs
`LTV_DUCKDB_PATH` exported and `--project-dir`/`--profiles-dir` pointed at `dbt/`, which is exactly
the plumbing this command exists to remove.

**dbt runs in-process, not as a subprocess.** That is what lets `profiles.yml` read configuration
from `Settings` through `env_var()`, but it also means dbt's DuckDB handle must be released when it
finishes or the next stage cannot open the warehouse. `transform.py` does this; do not remove it.

## 4. Environment notes

Portable constraints and the reasoning behind them. Two are worth understanding before changing
anything, because both shaped dependency choices that otherwise look arbitrary.

- **TLS-intercepting middleboxes.** Corporate proxies and some endpoint-security products present
  certificates signed by a locally-installed root. Tools that ship their own CA bundle then fail with
  `invalid peer certificate: UnknownIssuer`, even though the machine trusts the certificate.
  - Python: handled in code. `src/ltv/ingest/fetch.py` builds its SSL context with `truststore`, so
    verification uses the OS trust store. **Verification is never disabled — `verify=False` is banned
    in this repo.**
  - uv: set `UV_SYSTEM_CERTS=1` (`UV_NATIVE_TLS` is the deprecated spelling as of uv 0.12).
  - Node: set `NODE_EXTRA_CA_CERTS` to a PEM of your system roots if Evidence fails to fetch native
    binaries. npm generally works without it.
  - If `uv sync` fails with `os error 5 / access denied` on a cache rename, that is on-access
    virus scanning holding the temp file. It is transient — re-run.
- **The project must build without a C compiler.** PyTensor compiles model graphs to C and falls back
  to a much slower pure-Python backend when no compiler is present (`config.cxx == ''`, plus a
  `g++ not detected` warning). `numba` and `nutpie` are therefore pinned: both are prebuilt wheels
  needing no compiler and no admin rights, and nutpie gives a fast NUTS sampler regardless. Linux CI
  and the Docker image *do* have gcc, so the pipeline is exercised on both backends — which is a
  portability guarantee, not just a workaround. Set `PYTENSOR_FLAGS=cxx=` to silence the warning.
- **Node 22 LTS, not Node 24**, deliberately. Evidence pulls native DuckDB bindings, and where no
  compiler is available a prebuilt binary must exist for the Node ABI. Node 22 has far better
  prebuild coverage than 24.
- **brucehardie.com rejects the default `python-httpx` user agent** with a non-standard HTTP 465. The
  fetcher sends an honest self-identifying agent naming the project and its repo. Do not change this
  to impersonate a browser.
- **`make` is not assumed.** Every documented command is a plain `uv run ...` invocation.

## 5. Conventions

- **No hardcoded paths anywhere.** Every path derives from `Settings.repo_root` in
  `src/ltv/config.py`. This is what lets Docker relocate the tree with one env var. A hardcoded path
  in a diff is a blocking review comment.
- **No credentials in the repo.** There are none to have — all data sources are public downloads.
- **dbt layers:**
  - `staging/` — one model per source, rename/cast/clean only, no joins. Named `stg_<source>__<entity>`.
  - `intermediate/` — business logic; all cross-source unioning happens here. Named `int_<entity>__<verb>`.
  - `marts/` — consumption-ready. **Never reads `raw` or `source()`.** Named `dim_` / `fct_` / plain.
  - **Every model has at least one test.**
- **The staging contract.** Every `stg_*__transactions` model emits exactly:
  `source, customer_id, order_id, order_date, quantity, unit_price, gross_amount, is_return`.
  This is what makes adding a second source a config change rather than a rewrite.
- **Uniqueness is per `(source, customer_id)`, never `customer_id` alone.** Customer numbering is
  only unique within a dataset. A bare `unique` test on `customer_id` passes today and starts
  failing the moment a second source lands, so per-customer grain is asserted with a singular test.
- **No `dbt_utils`.** Every test this project needs is a few lines of plain SQL, and a readable
  singular test is better evidence of understanding than a macro call.
- Conventional commits, small and focused.
- Tests are real tests. A test that asserts `True` is worse than no test.

## 6. Modelling rules

These exist because getting them wrong produces a model that looks excellent and is worthless.

- **Features come only from the calibration window.** Nothing used to build calibration features may
  reference a date after the cutoff. Leakage here silently inflates every metric.
- **BG/NBD counts purchase occasions, not line items.** Same-day transactions for one customer collapse
  to a single occasion in the intermediate layer. Skipping this inflates frequency and flatters the model.
- **The four RFM quantities mean specific things, and three of them are commonly stated wrong.**
  These are implemented once, in `dbt/macros/rfm_summary.sql`:
  - `frequency` — **repeat** occasions, i.e. occasions minus one. A customer who bought once has 0.
  - `recency` — days between first and last occasion (age *at* last purchase). **Not**
    days-since-last-purchase, which is what marketing RFM means by the word; the two run in
    opposite directions.
  - `customer_age` — days from first occasion to the window end. "T" in Fader & Hardie notation,
    spelled out because DuckDB folds unquoted identifiers to lowercase.
  - `monetary_value` — mean spend over **repeat** occasions only, excluding the first purchase.
    That is the Gamma-Gamma convention; including the first purchase biases the estimate.
- **Holdout frequency counts every occasion, unlike calibration frequency.** The asymmetry is
  deliberate: the model predicts how many purchases come next — all of them, not all-but-one.
  Scoring a repeat count against a total count understates error by one purchase per active customer.
- **Ineligible customers are flagged, never filtered.** One-time buyers and zero-spend customers
  carry no Gamma-Gamma information, but they have valid BG/NBD frequencies and belong in customer
  counts. `is_gamma_gamma_eligible` marks them; the fit excludes them and reports how many.
- **MAP is the default fit** so the pipeline stays fast and CI stays honest. Any claim about
  *uncertainty intervals* must come from a `--full-bayes` (NUTS) run, not from MAP.
- **Report error against a naive baseline.** "The model beat nothing" is not a result.
- **LTV here means expected forward revenue over a stated horizon, undiscounted** — not gross margin.
  Margin and discount rate are business inputs these datasets do not contain. Say so in the README.

## 7. Definition of done

- [ ] One command from clean clone to populated warehouse and built dashboard
- [ ] Model output validated on a calibration/holdout split with reported error metrics
- [ ] Data quality tests that would actually catch a bad load
- [ ] A README a hiring manager can skim in 90 seconds: business question, architecture diagram, how
      to run it, what the results say
- [ ] A live dashboard URL

## 8. Honest status

Kept current. This section is what makes the repo credible — it must never overstate.

**Verified working:**
- Toolchain installed and pinned: Python 3.12.10, Node 22.23.2, uv 0.12.3.
- Single locked environment resolves dbt-core 1.12, Prefect 3.8.2, and PyMC-Marketing 0.19.4
  together — 195 packages, no conflicts.
- nutpie/numba sampling works on the compiler-less host.
- `ltv ingest cdnow` loads **69,659 rows / 23,570 customers** into `raw.cdnow_transactions`, matching
  the counts published in the dataset's own read_me. Asserted by tests, verified locally.
- Strict fixed-width parsing: field boundaries derived empirically (blank on all 69,659 lines at
  columns 0, 6, 15, 18), and any width, separator, encoding, or numeric deviation raises with the
  offending line number.
- 65 tests pass locally. The fast suite runs offline with no source data; the integration tests
  additionally require the downloaded file. On a clean clone they skip — except when `CI` is set,
  where a skip is escalated to a failure so a missing download can never leave a build green.
- `ltv transform` builds **8 dbt models and 66 dbt tests**, all passing.
- The occasion collapse works and reconciles: 69,659 line items become **67,591 occasions**
  (2,068 absorbed, 3.0%), and calibration (47,907) plus holdout (19,684) sums back to exactly 67,591.
- The window derivation lands on the canonical Fader & Hardie split with no rounding —
  calibration 1997-01-01…1997-09-30, holdout 1997-10-01…1998-06-30 — and is pinned by a test.
- RFM output matches values hand-computed in pandas, by independent logic, for three customers
  chosen to cover a one-time buyer, a mid-frequency buyer with a three-line day, and a heavy buyer
  with an eight-line day. A further test guards the *premise*: it fails if the sampled customers
  ever stop including multi-line-item days, which would leave the assertions passing while no longer
  covering the collapse at all.
- Calibration population: 23,570 customers, **14,119 one-time buyers** (~60%, consistent with the
  published CDNOW figures) and **16,512 who never returned in the holdout window**.
- Mutation-verified, Phase 1 and Phase 2 together: **10/10 mutations caught.** The checksum pin, the
  atomic cache write, the read-only connection guard, the `--force-download` wiring, the occasion
  grain, the lossless collapse, the leakage guard, the Gamma-Gamma monetary numerator *and*
  denominator, the window off-by-one, the holdout population, the holdout spend, the occasion
  amount, and the dbt connection release. Each was broken deliberately and the corresponding check
  confirmed to fail.

- **A whole-layer `dbt-modeler` review found five blocking defects, all in the tests rather than the
  models.** It verified the layer by independently recomputing all four RFM quantities for every one
  of the 23,570 customers straight from `raw` with separately-written SQL — 0 mismatches — and then
  found that the suite guarding them had holes. The worst: turning the `left join` in
  `int_customers__holdout_actuals` into an `inner join` drops the population from 23,570 to 7,058,
  removing precisely the customers the model predicts worst, and **passed all 62 tests**. A
  `relationships` test appeared to guard it but could not fail, because the model selects its
  `customer_id` out of the very model it was checked against. Also unguarded: holdout money entirely
  (`holdout_monetary_value` had no test and no description), the collapse in dollars as opposed to
  rows, and the `monetary_value` denominator. All four now have tests, and all four are in the
  mutation harness. See the four `assert_*` tests added in the same commit.

- **CI is green and genuinely exercised.** First run on 2026-08-13 executed the real ingest on Linux
  (69,659 / 23,570) and reported `49 passed` — not 43 passed with 6 skipped, confirming the
  integration tests actually ran rather than silently skipping. This also proves the pipeline works
  on PyTensor's C backend, not only the numba one used locally. Re-confirmed on the Phase 2 branch
  on 2026-08-18: `PASS=74` from dbt and `65 passed` from pytest, again with no skips. Note that CI
  triggers on `push: branches: [main]` and `pull_request` only, so a feature branch is exercised
  when its PR opens, not when it is pushed — a branch can sit for days looking untested because it
  genuinely is.
- Published at https://github.com/DomeNemeth/ltv-analytics-pipeline. `main` requires the `test` check
  to pass; `enforce_admins` is off, so the owner can still push directly.

**Known about the CDNOW master data, and what the layer does with it:** 255 byte-identical duplicate
rows and 80 `$0.00` rows are preserved verbatim in `raw` and deliberately **kept** through staging —
the file records line items with no order identifier, so duplicates are not provably artifacts, and
dropping them would understate spend by $4,332 across 164 customers. A test pins the count of 255 so
the decision gets revisited rather than silently reapplied if the source changes.

All 80 zero-value rows turn out to be **standalone occasions** — not one shares a day with a paid
purchase, so summing cannot rescue them. In the calibration window this leaves 14,120 customers
Gamma-Gamma-ineligible: 14,119 one-time buyers plus exactly **one** repeat buyer whose repeat spend
is zero. Worth stating plainly because the earlier estimate was wrong: the "68 zero-spend customers"
are almost all one-time buyers who were already excluded on that ground, so the zero-value rows cost
one additional customer, not 68.

**Not built yet:** model fitting, validation, marts, dashboard, Prefect flow, Docker, second data
source, published dashboard URL.

**Deliberate non-goals:** streaming/incremental loads, a warehouse other than DuckDB, multi-tenant
or scheduled production operation, margin-based LTV, customer-level PII handling (the datasets have none).

## 9. Subagents

Three agents live in `.claude/agents/`. Use them when a **fresh, focused context genuinely helps** —
a large diff, a whole-layer consistency sweep, a cross-file assumptions audit. Do not delegate small
or already-in-context work; do it inline and say the subagent was skipped and why.

- `dbt-modeler` — writes/reviews dbt models, tests, docs; owns naming, materialization, layer boundaries.
- `clv-validator` — statistical critic. **Must not write modelling code.** Audits assumptions, the
  calibration/holdout split, metric choice, and feature leakage. Must sign off before any results
  claim enters the README.
- `repo-reviewer` — pre-commit diff review against portfolio standards.

**Ask reviewers to verify by mutation, not by reading.** The Phase 1 review found three tests that
passed whether or not the code under test was correct — a `.partial`-file assertion satisfied by code
that never staged one, a read-only guard whose test short-circuited before reaching it, and a
`--force-download` seam tested at both ends but not in the middle. Reading the code found none of
them; deleting the logic and re-running the test found all three. A test that cannot fail is worse
than no test, because it buys false confidence.

`scripts/mutation_check.py` is now the harness for this, added in Phase 2 once it was clear the need
was recurring. It applies one deliberate defect at a time, runs `ltv transform` and `pytest`,
restores the file, and reports which guard noticed. **10 mutations, 10 caught.** Add a mutation
whenever a test claims to protect something load-bearing. Running it corrected a belief that reading
could not have: the weakened-occasion-grain mutation is caught by the `unique` test on
`occasion_id`, **not** by `assert_occasions_collapsed`, which needed its own mutation to prove it
can fail at all.

**The Phase 2 review sharpened the lesson: look hardest at the tests on the models you trust most.**
All five blocking findings were defects in the test suite, not in the SQL — the models recomputed
correctly for all 23,570 customers under independent arithmetic. The pattern worth internalising is
that a test asserting a property the model *establishes by construction* cannot fail. The
`relationships` test on `int_customers__holdout_actuals` checked an id that the model selects out of
the very relation it was checked against. When reviewing a test, ask what would have to change for
it to go red, and if the answer is "nothing reachable", it is decorative. A second pattern from the
same pass: a test that joins with `inner join` cannot notice missing rows, because the rows that
would fail it are the rows the join already dropped.

**Gotcha:** agents in `.claude/agents/` are loaded at session start, **from the working directory the
session was started in**. Opening a session in the parent folder rather than the repo means the
project's agents are not registered at all, and neither is a file created mid-session. In both cases,
use a general-purpose agent with the brief inlined — that is what the Phase 2 review pass used.

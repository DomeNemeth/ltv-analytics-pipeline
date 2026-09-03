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
uv run ltv fit                # fit BG/NBD + Gamma-Gamma, write model.customer_predictions
uv run ltv fit --full-bayes   # same, but NUTS instead of MAP -- the only run with real intervals
uv run ltv validate           # score against the holdout window, write reports/validation_*.md
uv run ltv validate --compare-models   # also fit MBG/NBD and Pareto/NBD on the same frame
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

**The dbt DAG is built in two halves, and `ltv transform` only builds the first.** Models tagged
`post_fit` — the staged predictions and `int_customers__scored` — read `model.customer_predictions`,
which does not exist until `ltv fit` has run. A build that did not select anything specific
therefore excludes them, so `ltv transform` works on a clean clone; `ltv validate` builds them with
`--select tag:post_fit` immediately before reading the result. The exclusion is applied in
`_excluding_post_fit`, not in `DEFAULT_ARGS`, so that `run_dbt(["build"])` and `run_dbt(None)` mean
the same thing — the first version put it in `DEFAULT_ARGS` and an explicitly-named `build` ran the
whole DAG, which broke every integration fixture. Any explicit `--select`/`--exclude` wins.

The real pipeline order is therefore **ingest → transform → fit → validate**, and Phase 6's flow is
that sequence.

**dbt runs in-process, not as a subprocess.** That is what lets `profiles.yml` read configuration
from `Settings` through `env_var()`, but it also means dbt's DuckDB handle must be released when it
finishes or the next stage cannot open the warehouse. `transform.py` does this; do not remove it.
Phase 4 made this load-bearing rather than theoretical: `ltv validate` invokes dbt and then opens the
warehouse read-only in the same process.

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
  needing no compiler and no admin rights, and nutpie gives a fast NUTS sampler regardless. Set
  `PYTENSOR_FLAGS=cxx=` to silence the warning.
  - **Nothing in this project currently runs on PyTensor's C backend, including in CI.** An earlier
    version of this section claimed Linux CI exercised it and that this was a portability
    guarantee. It is not true: `ltv fit` selects numba explicitly (below), so CI runs the same
    backend the development host does. CI's PyTensor cannot even link BLAS — it is a pip install,
    not conda — and PyTensor's own warning there recommends numba as the remedy. The portability
    claim that *is* supported is weaker and worth stating accurately: the fit produces identical
    customer counts and prediction row counts on Windows and on Linux.
  - **The Python fallback is not merely slower, it is unusable, and `ltv fit` sets the numba backend
    itself because of it.** Measured on this project's own data: the BG/NBD MAP fit on CDNOW takes
    **265 seconds** on the Python backend and **25** under numba, with parameter estimates identical
    to four decimal places. That is the difference between a command someone runs and a command they
    stop running. `_use_numba_backend()` in `src/ltv/models/clv.py` sets it in code rather than
    relying on an environment variable, so a fresh clone behaves like CI does.
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
- **Python-written relations enter the DAG the same way source data does.** `model.customer_predictions`
  is a dbt `source`, staged by `stg_model__customer_predictions`, and consumed from there. This is why
  the marts rule above needed no amendment when Phase 4 arrived and Phase 5's marts need to read
  predictions: they read `int_customers__scored`, not the `model` schema. A source read directly from
  a mart would have been the shortcut, and it is the one the rule exists to forbid.
- **Anything reading the `model` schema is tagged `post_fit`**, including its tests — explicitly, in
  each node's own config, rather than relying on dbt's indirect selection to infer it. See §3.
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
- [x] Model output validated on a calibration/holdout split with reported error metrics —
      `ltv validate`, scored against four naive baselines and two challenger models, and signed off
      by `clv-validator` on re-audit after it withheld sign-off once. Two conditions travel with the
      sign-off: no `--full-bayes` interval number reaches the README without its one-off /
      non-reproducible label, and nothing may present those HDIs as a range a customer's revenue
      will land in.
- [x] Data quality tests that would actually catch a bad load — not asserted, demonstrated: every
      guard in the suite has a deliberate defect in `scripts/mutation_check.py` that it is confirmed
      to catch.
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
- `ltv transform` builds **8 dbt models and 67 dbt tests**, all passing.
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
  integration tests actually ran rather than silently skipping. Re-confirmed on the Phase 2 branch
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

**Phase 3 — the CLV fit:**

- `ltv fit` runs in **58 seconds** on the compiler-less development host at MAP, inside the
  one-minute budget the plan set — but only because the fit selects the numba backend itself. On
  PyTensor's Python fallback the same fit takes 265 seconds. See §4.
- Fits **23,570 customers** with BG/NBD and **9,450** with Gamma-Gamma, excluding **14,120** who have
  no repeat spend to learn from. That exclusion count matches the dbt `is_gamma_gamma_eligible` flag
  exactly — the model and the dashboard cannot disagree about who exists.
- Writes **47,140 rows** to `model.customer_predictions`: every customer at both horizons, 273 days
  (the holdout length, read from the warehouse so it cannot drift from the scoring window) and 365
  days (the headline LTV figure). `horizon_days` is a column, so a revenue number can never be read
  without the window it applies to.
- Excluded customers receive the fitted **population** spend estimate rather than a NULL or a zero,
  flagged as `spend_estimate_source = 'population_mean'`. They are 60% of the customer base, so
  NULLing them would leave every dashboard total silently covering the other 40%.
- BG/NBD estimates on the calibration window: `r = 0.250`, `alpha = 33.0`, `a = 0.717`, `b = 2.125`.
  Recorded here so Phase 4 can compare them against the published CDNOW figures **read from the
  source paper rather than recalled** — and note the unit conversion that comparison needs, since
  the published work is in weeks and this project works in days, which rescales `alpha` by 7 while
  leaving `r` alone. No benchmark claim until that check is actually done.
- Expected spend per purchase: **35.83** population mean, **35.86** average across all customers.
  Total predicted forward revenue is **$642,283** over 273 days and **$815,143** over 365.
- `probability_alive` is within [0, 1] and interval columns are NULL on every MAP row. (An earlier
  version of this list also claimed forward revenue was verified to equal purchases times value.
  It does, and the claim was worthless: `predict()` computes it with that multiplication, so no
  reachable defect could make it fail. Removed under §9's own test.)
- CI runs the fit on the full population on Linux and produces identical counts to the Windows
  development host: 23,570 fitted, 9,450 in the spend model, 14,120 excluded, 47,140 rows. That is a
  cross-platform reproducibility check. It is **not** a backend portability check — see §4.
- Mutation-verified: fitting on `int_customers__rfm_full` instead of the calibration summary,
  swapping `recency` for `customer_age` in the `T` rename, fitting Gamma-Gamma on customers with no
  repeat spend, and dropping the population fallback. All four were broken deliberately and caught.
  None is visible to the dbt suite.

**The 14% under-prediction, diagnosed.** At 273 days the model predicts **16,867** purchases against
**19,684** observed. `clv-validator` established what this is and is not:

- **Not a MAP or prior artefact.** An independent re-implementation of the BG/NBD likelihood,
  maximised with no priors at all, lands 0.17 log-likelihood units from our MAP over 23,570
  observations and predicts 16,874. `--full-bayes` will produce intervals; it will not move the
  point forecast.
- **Not an apples-to-oranges comparison.** Every one of the 23,570 customers was first seen between
  1997-01-01 and 1997-03-25, so no holdout occasion is anyone's first purchase and BG/NBD's repeat
  count is the right comparand. The horizon lines up to the day.
- **It is real misspecification: the process is not stationary.** Monthly repeat occasions fall ~40%
  through calibration (3,690 → 2,220) and then *plateau* at ~2,200 through the holdout rather than
  continuing down. BG/NBD can only explain the calibration decline as dropout plus heterogeneity
  sorting, so it extrapolates a decay the real cohort stops doing. In-sample fit is excellent
  (+0.8% on calibration repeat transactions); the error is entirely out-of-sample and grows with
  horizon (ratio 0.945 at 30 days → 0.857 at 273).
- **The shortfall is concentrated:** customers with exactly 1 or 2 calibration repeats account for
  1,940 of the 2,817 missing purchases (69%). BG/NBD writes them off too aggressively — 37.8% of the
  one-repeat group came back.

Phase 4 should fit MBG/NBD or Pareto/NBD on the same frame and compare that bucket table. Either
outcome is a stronger README line than a bare error number.

**Phase 4 — validation:**

- `ltv validate` scores **23,570 customers** over the 273-day holdout window and writes
  `reports/validation_cdnow.md`, three charts, and **98 rows** to `model.validation_metrics` (fewer
  without `--compare-models`, which is what CI runs and what the committed report reflects). It
  builds the post-fit dbt models itself immediately before reading them, so a report can never
  describe a relation built from an earlier state of the warehouse.
- **The model beats four naive baselines on aggregate error, and the margin is smaller than the
  first version of this section claimed.** Predicted 16,867 purchases against 19,684 actual
  (**−14.3%**). The naive rules: same-as-last-window **24,337 (+23.6%)**, calibration-rate
  **29,011 (+47.4%)**, population-mean **29,001 (+47.3%)**.

  **The two rate baselines are inflated and their +47% should not be quoted as the comparison.**
  They divide by each customer's own observation length and multiply by the holdout length; mean
  `customer_age` is 229 days against a 273-day holdout, so they scale every calibration count up by
  1.19 before comparing. Roughly half of that +47% is the rescaling, not naivety. The honest
  comparand is the same-period carry-forward rule, which needs no rescaling because both windows are
  39 weeks to the day: **−14.3% against +23.6%**. Still a clear win, and a much more modest one
  than the number this section carried before an audit caught it.
- **Per-customer MAE is not evidence of much, and the report now says so.** MAE 0.818 against
  0.911 / 1.038 / 1.389 for the naive rules — but **predicting that nobody buys anything at all
  scores 0.835**, because 16,512 of 23,570 customers genuinely buy nothing. On a quantity that is
  70% zeros the all-zero rule *is* the MAE floor, and the model clears it by **2.0%**. Every naive
  rule above is worse than doing nothing. This model's value is aggregate and ordinal; it is not
  per-customer accuracy, and any README line implying otherwise is unsupported.
- **Fitting is not predicting, quantified.** In-sample the model expects 24,533 calibration repeat
  transactions against 24,337 actual (**+0.8%**); out-of-sample it is **−14.3%**. Both are printed
  in the report, adjacent and labelled, because the in-sample figure is the one that would otherwise
  get quoted as accuracy.
- **Ranking works, and almost all of it is available without the model.** The top decile by
  predicted forward revenue captures **51.6%** of realised holdout revenue against 10% for a
  ranking with no discriminating power — but ranking the same customers by the naive rule captures
  **50.4%**, within 1.1 points. On rank correlation the naive rule is actually **ahead**: Spearman
  ρ **0.479 against the model's 0.421** on revenue and **0.485 against 0.442** on counts.

  So "which segments are worth acquiring" is largely answered by sorting customers on what they
  already spent. What the model adds is *calibration* rather than *order* — a scale that is 14% low
  rather than 47% high, and a value for every customer including those with no repeat history. For
  choosing who to target the naive rule is competitive; for forecasting how much, it is not. The
  earlier version of this bullet claimed the ranking as a model win against a straw 10% comparator
  and had no baseline in that section at all.
- **MAPE is not reported on purchase counts**, deliberately: 16,512 of 23,570 actuals are zero, so
  it would be computed over 30% of the population and read as describing all of it. Where MAPE is
  reported it carries its exclusion count.
- **Gamma-Gamma barely beats "past average order value is future average order value".** On the
  7,058 customers who returned: MAE 18.14 against the naive rule's 19.04, aggregate error −2.6%
  against −2.7%, MAPE 63.4% against 64.8%. The spend model earns its place by giving *every*
  customer an estimate including the 14,120 with no repeat spend, and by carrying uncertainty into
  the revenue product — not by being much more accurate per customer than arithmetic. Stated
  plainly because the table would otherwise be read as validating the model choice, and it does not.
  It is also consistent with the independence violation below: shrinking everyone toward a common
  mean is what limits it.

**The challengers settle the misspecification question.** MBG/NBD and Pareto/NBD were fitted at MAP
on the identical calibration frame, and the answer is not the one the Phase 3 notes guessed at:

- **Pareto/NBD is materially better out of sample: −7.5% against BG/NBD's −14.3%.** In the x=1 and
  x=2 buckets, where 69% of the shortfall lives, it misses **749** purchases against BG/NBD's
  **1,940**. Per-customer MAE is near-identical across all three models (0.795–0.818), which is
  itself the finding: aggregate error separates them and individual error does not.
- **MBG/NBD has the best per-customer MAE of all seven predictors and the worst aggregate error.**
  0.795, clearing the all-zero floor by 4.8% against the champion's 2.0% — while landing −19.3% on
  the total. That single row is the clearest statement in the project of why both kinds of metric
  have to be reported: picked on MAE it wins, picked on aggregate error it comes last, and neither
  choice is wrong on its own terms.
- **On aggregate error MBG/NBD is worse than the champion (−19.3%).** Removing BG/NBD's "no repeat, therefore still
  alive" assumption makes the zero-repeat group *worse*, not better — predicted 0.182 against 0.251
  actual, where BG/NBD manages 0.230. That was the opposite of the expected result and is worth
  stating plainly.
- So the 14% gap is a **dropout-mechanism** problem, not an estimation problem: letting a customer
  churn at any moment beats letting them churn only just after a purchase. Fits took 7s (MBG/NBD)
  and 20s (Pareto/NBD) on the compiler-less host — the Pareto/NBD `hyp2f1` risk did not materialise,
  because the challengers go through the same numba backend selection the champion does.
- **Nothing was promoted.** `model.customer_predictions` is still BG/NBD. Changing the champion on
  this evidence is a decision to take deliberately in a later phase, not a side effect of measuring.

**Against the published CDNOW figures, read from the paper rather than recalled.** Fader, Hardie &
Lee (2005), Table 2, p. 281. Converting `alpha` from days to weeks (÷7; `r`, `a`, `b` are
dimensionless and unconverted):

| | r | alpha (weeks) | a | b |
|---|---|---|---|---|
| This project, 23,570 customers | 0.250 | 4.716 | 0.717 | 2.125 |
| Published, 2,357 customers | 0.243 | 4.414 | 0.793 | 2.426 |

This is a **consistency check, not a reproduction** — the paper fits a 1/10th systematic sample and
this fits the full master file. Two independent details do reproduce: the zero-repeat class is 59.9%
of the population in both (1,411/2,357 and 14,119/23,570). The paper's headline "under-forecasting
by 4%" is a **cumulative** figure across all 78 weeks and must not be read against a holdout-only
error; the report gives both bases so the comparison is like for like.

**Two assumption violations the README must disclose.** Both were measured, not assumed, and both
are now reproduced by `ltv validate` on every run rather than by a one-off script:

- **Gamma-Gamma's frequency/monetary independence does not hold here.** Spearman ρ = **+0.198**
  (p = 3e-84) across the 9,450 eligible customers, and mean repeat order value climbs monotonically
  with frequency from 33.42 at one repeat to 44.93 at eleven or more — a 34% spread. (The
  committed report buckets at `7+` to match the published CDNOW breakdown, so it shows the same
  climb ending at 42.71, a 28% spread. Same data, different top bucket; neither figure is wrong and
  the report is the one that reproduces.) The model
  shrinks toward a common population mean, so it systematically under-values heavy buyers and
  over-values light ones. That is the exact axis a "which segments are worth acquiring" conclusion
  runs along, so segment-level LTV rankings are compressed.
- **`probability_alive` is exactly 1.0 for 14,119 customers — 59.9% of the base — by construction.**
  BG/NBD only allows dropout immediately after a purchase, so a customer with zero repeats has never
  had an opportunity to drop out. Only 14.6% of them actually transacted in the holdout window. This
  is a model tautology, not a finding: no segment definition or dashboard tile may treat it as
  evidence that these customers are healthy.

**Outliers are not a problem here**, checked so it does not get re-litigated: max repeat order value
is 756.47, the 99.9th percentile is 300.63, and the top 10 customers hold 1.24% of eligible repeat
spend. Trimming barely moves the population mean (35.83 untrimmed, 34.41 dropping the top 1%).

**A build-hygiene failure worth remembering, because it produced wrong numbers behind a clean
`git diff`.** `scripts/mutation_check.py` restored the source file after each mutation but never
rebuilt the warehouse, so a run ending on `monetary-denominator` left
`int_customers__rfm_calibration` **materialised from mutated SQL** — `monetary_value` divided by
occasions instead of repeat occasions, about a third too low. The next `ltv fit` read it, fitted
Gamma-Gamma on corrupted spend, and wrote a predictions table that looked entirely normal: positive,
smaller than total spend, correctly zero for one-time buyers. Nothing in the repo could see it. Every
Python test builds its own synthetic frame, and the dbt test that does catch it only runs on a
rebuild. Two fixes, both in place: the harness now rebuilds in its `finally` so no mutation outlives
its run, and `check_assumptions` verifies its own inputs — `monetary_value * frequency` must equal
`total_spend - first_occasion_spend`, which is why the macro now exposes `first_occasion_spend`.
**The lesson generalises: restoring source is not restoring state.**

**`--full-bayes` has now been run on the full CDNOW population**, so the repo finally contains
uncertainty produced from real data rather than from synthetic tests.

**Read every figure in this block as a one-off run, not as a reproducible artifact.** MAP is the
committed champion (§6), so `ltv fit` was re-run afterwards and the warehouse now holds `map`
predictions with NULL intervals. Nothing committed reproduces the numbers below — an audit checked
and correctly could not verify them. To see them again, run `uv run ltv fit --full-bayes`. They are
recorded here because the run happened and the results matter; they are labelled because a status
document that reads as though a plain `ltv validate` produces them would be lying by implication.

- **6 minutes 4 seconds** at NUTS on the compiler-less host, against 29 seconds at MAP. Same
  population every step of the way: 23,570 fitted, 9,450 in the spend model, 14,120 excluded,
  47,140 rows. `Settings.random_seed` now reaches something for the first time.
- **The point forecast did not move.** 16,860 expected holdout purchases against MAP's 16,867
  (0.04%), and $641,905 forward revenue against $642,283 (0.06%). Both round to the same −14.3%.
  This confirms on real data what an independent no-prior re-implementation predicted during the
  Phase 3 audit: sampling buys intervals, not a different answer. The misspecification is not an
  estimation artefact and cannot be sampled away.
- **All 47,140 rows now carry a 94% HDI**, where every MAP row is NULL.

**What those intervals are, stated carefully, because the narrow numbers invite a serious
misreading.** They are intervals on the *model's expectation* for a customer, not on what that
customer will do. With 23,570 observations the four BG/NBD parameters are pinned down tightly, so
the intervals are correspondingly tight: **6-10% of the estimate**. The highest-value customer is
estimated at $6,111 with an HDI of [$5,881, $6,318] — which reads like near-certainty about one
person's future spending and is nothing of the sort.

Measured rather than argued: **0.6% of customers' realised holdout spend falls inside their own 94%
HDI.** That is not a broken interval. An individual outcome is dominated by the Poisson and gamma
variation the parameter posterior deliberately excludes, and a customer whose expectation is $6,111
can perfectly well spend $0. It does mean **no dashboard tile and no README line may present these
as a range a customer's revenue is likely to land in.** A genuine predictive interval would need the
posterior *predictive* distribution, which this project does not currently compute.

**MAP remains the default and the committed champion**, per §6: the pipeline stays fast, CI stays
honest, and every number in the committed report comes from a deterministic fit.

**The stale-predictions guard, and why it needed to exist.** `ltv fit` now writes `model.fit_runs`
recording summary statistics of the frame it actually trained on, and
`assert_predictions_match_the_current_calibration_inputs` recomputes those from the warehouse and
fails the post-fit build if they disagree. This closes deferred finding A2, and the warehouse was
sitting in exactly the failure state when Phase 4 began — dbt models rebuilt at 00:16, fit artifacts
from 23:51, nothing anywhere able to say the predictions no longer matched their inputs. The
recomputation is deliberately independent SQL rather than a shared helper: a test that shares logic
with the thing it checks cannot fail (§9).

Note it compares **content, not timestamps**. Re-running `ltv transform` after a fit rebuilds the
same numbers and the guard stays quiet, which is the point — a timestamp-based check would cry wolf
on every rebuild and get ignored within a week.

**A second instance of "restoring source is not restoring state", found by looking at a chart.**
`reports/` is tracked, and `ltv validate` writes into it. A mutation-testing run therefore leaves
committed charts and a committed report **computed from mutated code** — the source is clean, the
diff of source is clean, and `reports/cdnow_revenue_by_decile.png` is quietly wrong. This was caught
by opening the PNG and noticing decile 1's actual revenue was exactly half the value in the report
table beside it, which is the `holdout-spend` mutation's signature. `_rebuild_state` in the harness
now regenerates reports as well as the warehouse, and CI diffs `reports/` after regenerating them so
a report that does not reproduce fails the build.

**Not built yet:** marts, dashboard, Prefect flow, Docker, second data source, published dashboard
URL.

**Deliberate non-goals:** streaming/incremental loads, a warehouse other than DuckDB, multi-tenant
or scheduled production operation, margin-based LTV, customer-level PII handling (the datasets have none).

## 9. Subagents

Three agents live in `.claude/agents/`. Use them when a **fresh, focused context genuinely helps** —
a large diff, a whole-layer consistency sweep, a cross-file assumptions audit. Do not delegate small
or already-in-context work; do it inline and say the subagent was skipped and why.

- `dbt-modeler` — writes/reviews dbt models, tests, docs; owns naming, materialization, layer
  boundaries. Ran its whole-layer pass on Phase 2; findings and fixes are in §8.
- `clv-validator` — statistical critic. **Must not write modelling code.** Audits assumptions, the
  calibration/holdout split, metric choice, and feature leakage. Must sign off before any results
  claim enters the README. Ran its first audit on the Phase 3 fit and **withheld sign-off** — it
  found the corrupted-warehouse fit, diagnosed the 14% gap as non-stationarity rather than an
  estimation problem, and measured two assumption violations (§8). All are now fixed or disclosed.
  It is still owed the Phase 4 metrics audit, and must sign off before any accuracy claim reaches
  the README.
- `repo-reviewer` — pre-commit diff review against portfolio standards.

**Ask reviewers to verify by mutation, not by reading.** The Phase 1 review found three tests that
passed whether or not the code under test was correct — a `.partial`-file assertion satisfied by code
that never staged one, a read-only guard whose test short-circuited before reaching it, and a
`--force-download` seam tested at both ends but not in the middle. Reading the code found none of
them; deleting the logic and re-running the test found all three. A test that cannot fail is worse
than no test, because it buys false confidence.

`scripts/mutation_check.py` is now the harness for this, added in Phase 2 once it was clear the need
was recurring. It applies one deliberate defect at a time, runs `ltv transform` and `pytest`,
restores the file, and reports which guard noticed. **21 mutations, 21 caught.** Add a mutation
whenever a test claims to protect something load-bearing. Six of the twenty-one break Python rather
than SQL: the dbt suite is blind to all of them, because every one produces a model that fits,
converges, and reports plausible numbers.

Since Phase 4 the harness runs `ltv validate` as well as `ltv transform` and `pytest`, because
`ltv transform` excludes `tag:post_fit` and would therefore be blind to every mutation in the
scoring layer. Three of the Phase 4 mutations are caught by `ltv validate` **alone** — nothing else
in the repo notices them. A mutation can also opt into a more expensive check by name:
`stale-fit` needs `ltv fit` to run, because the provenance row it corrupts is only written at fit
time and a validate against the previous run's honest row would pass. Running it corrected a belief that reading
could not have: the weakened-occasion-grain mutation is caught by the `unique` test on
`occasion_id`, **not** by `assert_occasions_collapsed`, which needed its own mutation to prove it
can fail at all.

**Phase 4 added a lesson about the mutations themselves, not the tests.** The obvious defect to
plant in `int_customers__scored` was turning a `left join` into an `inner join` — the exact shape of
the Phase 2 disaster. It **survived the whole harness**, and the guard was not at fault: upstream,
`assert_customer_populations_align` already pins the holdout and calibration populations to the same
keys, so at that join the two forms are genuinely equivalent and no observable behaviour changes.
The mutation was inert. Replacing it with a `where is_gamma_gamma_eligible` filter on the
calibration CTE — a plausible edit that drops 14,120 of 23,570 customers — produced an immediate
catch. So: **a surviving mutation means one of two things, and they need telling apart.** Either the
guard is decorative, or the defect was not a defect. Assuming the first and rewriting a working test
is its own way of making the suite worse.

**Also from Phase 4: the harness poisons more than the warehouse now.** `ltv validate` writes into
`reports/`, which is tracked, so a mutation run leaves committed charts and a committed report
computed from broken code behind a clean source diff. `_rebuild_state` regenerates them; CI diffs
`reports/` after regenerating. The pattern has now bitten twice in two different places, which is
what makes it a rule rather than an anecdote.

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

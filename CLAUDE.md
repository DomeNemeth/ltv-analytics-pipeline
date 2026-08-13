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
uv run pytest                 # Python tests; integration tests skip if source data is absent
uv run ruff check . && uv run ruff format --check .
```

Commands are added to the CLI as each stage lands, so `uv run ltv --help` always reflects what the
repo can actually do. It never advertises stubs.

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
- Conventional commits, small and focused.
- Tests are real tests. A test that asserts `True` is worse than no test.

## 6. Modelling rules

These exist because getting them wrong produces a model that looks excellent and is worthless.

- **Features come only from the calibration window.** Nothing used to build calibration features may
  reference a date after the cutoff. Leakage here silently inflates every metric.
- **BG/NBD counts purchase occasions, not line items.** Same-day transactions for one customer collapse
  to a single occasion in the intermediate layer. Skipping this inflates frequency and flatters the model.
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
- 49 tests pass locally. The fast suite (43) runs offline with no source data; the 6 integration
  tests additionally require the downloaded file and are the ones asserting the row/customer counts.
  On a clean clone the integration tests skip — except when `CI` is set, where a skip is escalated to
  a failure so a missing download can never leave a build green.
- The checksum pin, the atomic cache write, the read-only connection guard, and the
  `--force-download` wiring are each **mutation-verified**: the logic was removed and the
  corresponding test confirmed to fail.

**Written but never executed:** `.github/workflows/ci.yml`. There is no git remote yet, so CI has
never run even once. Do not describe the build as passing until it has actually run.

**Known about the CDNOW master data, unresolved by design until staging:** 255 byte-identical
duplicate rows, 80 rows with `$0.00`, and 1,774 customer-days holding more than one row (collapsing
to purchase occasions removes 2,068 rows, 3.0%). All are preserved verbatim in `raw`.

**Not built yet:** dbt project, model fitting, validation, dashboard, Prefect flow, Docker, second
data source, GitHub remote, published URL.

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
than no test, because it buys false confidence. `scripts/` has no mutation harness yet — the Phase 1
one was throwaway; write one if this becomes routine.

**Gotcha:** agents in `.claude/agents/` are loaded at session start. One created mid-session cannot be
invoked by name until the next session; until then, use a general-purpose agent with the brief inlined.

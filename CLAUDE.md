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
| Sampling | nutpie + numba | The development host has **no C compiler**, so PyTensor cannot use its C backend. See §4. |
| Orchestration | Prefect | Earns its place for exactly one reason: the pipeline is `dbt → Python → dbt` (RFM features feed the model fit, whose predictions feed the marts). That handoff is awkward in pure dbt. |
| Dashboard | Evidence.dev | Builds a static site, so it publishes free to GitHub Pages. Streamlit would need a running server. |
| Packaging | uv | Fast, lockfile-based, pins the interpreter. |

**Prefect is capped at one flow file.** No deployments, no work pools, no agents, no Prefect Cloud.
If it grows past `flows/ltv_pipeline.py`, it has become a framework showcase and should be cut back.

**Ingestion is deliberately thin.** No Parquet archive layer, no incremental loading, no source
freshness scheduling, no connector abstraction. That ground is covered by a previous project; the
budget here goes to modelling and validation.

## 3. Commands

Everything runs through `uv`; there is no `make` on the development host.

```bash
uv sync                       # install the locked environment
uv run ltv info               # show resolved config and whether the warehouse is built
uv run pytest                 # Python tests
uv run ruff check . && uv run ruff format --check .
```

Commands are added to the CLI as each stage lands, so `uv run ltv --help` always reflects what the
repo can actually do. It never advertises stubs.

## 4. Environment gotchas (development host specific, not in the repo's public docs)

These are real, already-diagnosed, and will waste hours if rediscovered from scratch.

- **Some endpoint security products intercept TLS.** Tools that ship their own CA bundle fail with `invalid peer certificate:
  UnknownIssuer`. Fixes in place:
  - `UV_SYSTEM_CERTS=1` (user env var). Verified: without it uv cannot reach PyPI, with it resolution
    succeeds. Note `UV_NATIVE_TLS` is the deprecated spelling as of uv 0.12.
  - `NODE_EXTRA_CA_CERTS=<local path>` (user env var), additive
    insurance for when Evidence pulls native binaries. npm itself works without it.
- **`uv sync` occasionally fails with `os error 5 / access denied`** on a cache rename. That is
  on-access virus scanning holding the temp file. It is transient — just re-run.
- **No C compiler on the host** (no MSVC Build Tools, no mingw). PyTensor reports `g++ not detected`
  and `config.cxx == ''`, falling back to its slow Python backend. Hence `numba` + `nutpie`:
  both are prebuilt wheels needing no compiler and no admin. Verified working — nutpie NUTS recovers
  a known mean correctly in ~9s. Set `PYTENSOR_FLAGS=cxx=` to silence the warning.
  Docker/CI run on Linux with gcc present, which also proves the pipeline is backend-portable.
- **Toolchain is installed per-user, no admin**, via direct downloads (system package managers were unavailable
  by the same TLS problem): Python 3.12.10 (`py -3.12`), Node 22 LTS at
  `<local path>` (portable zip, not the MSI — the MSI needs admin), uv
  at `<local path>`.
- **Node 22 LTS, not Node 24**, deliberately. Evidence pulls native DuckDB bindings, and with no
  compiler available a prebuilt binary must exist for the Node ABI. Node 22 has far better prebuild
  coverage than 24.

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
- `ltv info` runs; config tests pass.

**Not built yet:** ingestion, dbt project, model fitting, validation, dashboard, Prefect flow,
Docker, CI beyond lint+test, second data source, published URL.

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

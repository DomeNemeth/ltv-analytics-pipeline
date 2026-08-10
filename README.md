# Customer Lifetime Value Analytics Pipeline

> **Status: in development.** This README is a stub. The full write-up — business question,
> architecture diagram, results, and the live dashboard link — lands at the end of the build.
> See [CLAUDE.md](CLAUDE.md) for the current state of play and what is actually verified.

Transactional e-commerce data → DuckDB warehouse → dbt (RFM + cohorts) → probabilistic
lifetime-value predictions (BG/NBD + Gamma-Gamma) → published segment dashboard.

**The question it answers:** which customer segments are worth acquiring and retaining, and what is
each segment's expected forward revenue?

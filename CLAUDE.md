# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ganyan is a Turkish horse racing prediction system. It scrapes race data from TJK (Türkiye Jokey Kulübü), stores it in PostgreSQL, and generates Bayesian predictions served via CLI and Flask web app.

## Commands

```bash
# Start PostgreSQL
docker compose up -d

# Install dependencies
uv sync --all-extras

# Run database migrations
uv run ganyan db init

# Scrape today's race cards
uv run ganyan scrape --today

# Scrape today's results
uv run ganyan scrape --results

# Backfill historical data
uv run ganyan scrape --backfill --from 2024-01-01

# Predict a specific race
uv run ganyan predict <race_id>

# Predict all today's races
uv run ganyan predict --today
uv run ganyan predict --today --json

# List races
uv run ganyan races --today
uv run ganyan races --date 2024-03-15

# Start web app (port 5003)
uv run python -c "from ganyan.web.app import run; run()"

# Run tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_predictor/test_bayesian.py::test_probabilities_sum_to_100 -v
```

## Project goal (2026-05-03)

**Pick winners across all bet types.** Tek, İkili, Üçlü, 4'lü, 5'lı, 6'lı, 7'li — every structure. The model and pipeline exist to serve this goal, not to satisfy academic process rules. Hit rate is the primary metric; engineering work serves picking winners, not the other way around.

## Critical invariants (read before consulting /advice for a real bet)

1. **Model beats chance ≠ bets have edge.** The 2026-05-02 chance-hypothesis settlement proved the model has 3-17× lift over random (p≈0). The same date's OOS retest proved real kept-race ROI is **−20% to −30% on uclu_box6 and sirali_ikili_top1** — the takeout floor. These are independent claims. The advice gate filters for confidence, not edge after takeout.

2. **The halt flag is authoritative.** `/tmp/ganyan-halt.flag` (or `$GANYAN_HALT_FLAG_PATH`) is set by canaries (rolling-PnL, uniformity guard, heartbeat, scrape integrity, regime monitor). When set, `/advice` and `ganyan advice` suppress Kelly stakes. Manually clear with `rm /tmp/ganyan-halt.flag` only after investigating the reason.

3. **OOS bar: ≥365 days AND ≥1500 races.** Set by the V2 retraction (2026-05-02). Enforced in `logs/discordance_oos_backtest.py:assert_min_window`. Do not bypass.

4. **Frame around winning horse and winning bet, not payout/ROI** — primary metric is top-1 hit rate. (Existing memory; reaffirmed here.)

5. **Tek + Plase is the default ticket when model #1 ≥ 30%.** Established 2026-05-04 after a 13-race day where the model called 5 winners at 22-43% conviction we skipped to chase exotics, while Üçlü-K3 set match was only 15% (2/13) — even with 77% top-3 hit rate. Üçlü K-3 has takeout penalty without multiplier upside; high-multiplier exotics (4'lü/5'lı/6'lı/7'lı) and Tek+Plase are the structures with mathematical paths to edge. **Never recover losses with bigger exotics** — recovery comes from the next clean signal.

6. **Pull live ensemble before any bet recommendation.** `ganyan predict --today` invokes the single-head MLPredictor; the daemon's scheduled inference runs the 11-head EnsemblePredictor and writes to the `Prediction` table. These can disagree by 10pp. Query `Prediction` rows directly (sort `predicted_at desc`) before quoting probabilities for any stake-sizing decision. Re-pull within 30 minutes of post when stakes are >100 TL.

## Architecture

Three-layer service-oriented monorepo sharing PostgreSQL:

1. **Scraper** (`src/ganyan/scraper/`) — TJK website client using AJAX endpoints at `/TR/YarisSever/Info/Sehir/GunlukYarisProgrami`. `tjk_api.py` fetches race cards and results per city via `SehirId` parameters. `parser.py` normalizes raw HTML data into dataclasses. `backfill.py` handles idempotent storage and incremental historical loading.

2. **Predictor** (`src/ganyan/predictor/`) — Empirical Bayesian model. `features.py` extracts speed figure, form cycle (exponential decay), weight delta, rest fitness (Gaussian curve), and class indicator. `bayesian.py` computes prior (1/N) x feature likelihoods → normalized probabilities with confidence scores and contributing factors.

3. **Web + CLI** (`src/ganyan/web/`, `src/ganyan/cli/`) — Flask app with HTMX (Bootstrap 5, Turkish UI). Typer CLI for terminal use. Both consume predictor and scraper directly.

### Data Flow

```
TJK website (AJAX per city) → scraper/tjk_api.py → scraper/parser.py → scraper/backfill.py → PostgreSQL
                                                                                                    ↓
CLI (ganyan predict) ← predictor/bayesian.py ← predictor/features.py ← race_entries table
Flask (/races/<id>/predict) ←────────────────┘
```

### Key Turkish Racing Metrics

- **HP** — Handikap Puanı (handicap points)
- **KGS** — Koşmama Gün Sayısı (days since last race; 14-28 optimal)
- **S20** — Son 20 yarış performansı (last 20 races performance)
- **EİD** — En İyi Derece (best time, stored as string "1.30.45", converted to seconds for computation)
- **GNY** — Günlük Nispi Yarış puanı (daily relative race score)
- **AGF** — Ağırlıklı Galibiyet Faktörü (weighted win factor)

### Database

PostgreSQL 16 via Docker Compose. SQLAlchemy 2.0 ORM + Alembic migrations. Tables: `tracks`, `races` (unique on track+date+race_number), `horses` (unique on name), `race_entries` (pre-race + post-race fields in one row), `scrape_log`.

### Config

`pydantic-settings` reads from `.env` file or environment variables. See `.env.example`. Key: `DATABASE_URL`, `TJK_BASE_URL`, `SCRAPE_DELAY`, `FLASK_PORT`.

### Reporting Conventions

When discussing model or strategy performance, frame around the **winning horse** and **winning bet** — not the payout or ROI.

- **Primary metric**: top-1 hit rate (did we pick the winning horse?)
- **Secondary**: top-3 hit rate; binary strategy hit rate (`ganyan_top1`, `sirali_ikili_top1`, `uclu_top1`, `uclu_box6`)
- **Tertiary** (only when sizing strategy or explicitly asked): payout/ROI/net-TL

Payout reflects TJK pool dynamics (takeout, retail behavior, "devren" carryovers) more than model quality. A 33% top-1 day on short-priced favorites shows −25% ROI because the math doesn't work at 1.9× average odds — but the model is doing its job. Don't anchor model-quality reports on money.

## Maintenance routines

| Cadence | Task | How |
| --- | --- | --- |
| Daily 12:00 | Heartbeat (liveness + uniformity) | `launchctl list \| grep com.ganyan.heartbeat` |
| Daily 12:30 | Scrape integrity (AGF drift) | `launchctl list \| grep com.ganyan.integrity` |
| Daily 23:30 | Regime monitor (takeout drift) | `launchctl list \| grep com.ganyan.regime` |
| Weekly | Commit-ratio audit (ganyan vs linguistic) | `git log --since="7 days ago" --oneline \| wc -l` in each repo; if ganyan > 5× linguistic for 2 consecutive weeks, force a Ganyan freeze week |

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

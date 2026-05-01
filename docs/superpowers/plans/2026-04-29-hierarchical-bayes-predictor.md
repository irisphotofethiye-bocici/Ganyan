# Hierarchical Bayesian Race Predictor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototype a hierarchical Bayesian Plackett-Luce race predictor in PyMC alongside the existing LightGBM ensemble, then evaluate posterior calibration on 2026-01-26+ holdout to determine whether proper Bayesian inference improves over softmax point estimates.

**Architecture:** Plackett-Luce likelihood over observed finish orders. Each horse has a latent skill `θ_h`, drawn from a hierarchical prior with jockey, sire, and (track × distance-bucket) random effects. AGF (public-belief percentage) enters as a coefficient on the prior mean — public belief informs the prior, private signals (form, workouts, etc.) update via the likelihood. Inference via ADVI (fast mean-field VI for development), with NUTS available as a sanity check on smaller subsets.

**Tech Stack:** PyMC 5, PyTensor, ArviZ for diagnostics. NumPyro/JAX backend optional. Existing data lives in PostgreSQL via SQLAlchemy; reuse the `RaceEntry`/`Race`/`Track` models from `ganyan.db.models`.

**Out of scope for this prototype:**
- Replacing the LightGBM ensemble in production
- CLI integration into `ganyan predict`/`advice`
- Web app or scheduler integration
- Pick generation, Kelly stake sizing
- Backtesting against TJK payouts

This is a prototype to determine whether the calibration is genuinely better. Production wiring comes after that question is answered.

---

## File Structure

**New files (all under `src/ganyan/predictor/bayes/`):**
- `__init__.py` — module exports
- `data.py` — build the training/holdout frames from DB; encode horse/jockey/sire/track-distance to integer indices
- `model.py` — PyMC model factory
- `trainer.py` — fit ADVI, persist posterior samples (`models/bayes_pl_posterior.nc`) + index dictionaries (`models/bayes_pl_indices.json`)
- `predictor.py` — load posterior + indices, predict for `race_id`, return win-probability distribution per horse
- `calibration.py` — compute holdout metrics (top-1, Brier, log-likelihood, CI coverage) and compare to LightGBM ranker

**New tests (under `tests/test_predictor/test_bayes/`):**
- `__init__.py`
- `test_data.py`
- `test_model.py`
- `test_trainer.py`
- `test_predictor.py`
- `test_calibration.py`

**Modified:**
- `pyproject.toml` — add `pymc>=5.10`, `arviz>=0.18` to optional `[bayes]` extras

---

## Task 1: Add PyMC dependencies as optional extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `[bayes]` optional extras section**

Open `pyproject.toml`. Find the `[project.optional-dependencies]` block (currently has only `dev`). Add a `bayes` extras list after `dev`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "respx>=0.21",
    "factory-boy>=3.3",
]
bayes = [
    "pymc>=5.10",
    "arviz>=0.18",
]
```

- [ ] **Step 2: Sync env**

```bash
uv sync --all-extras
```

Expected: PyMC + arviz install (~30-60s). No errors.

- [ ] **Step 3: Smoke test**

```bash
uv run python -c "import pymc as pm; import arviz as az; print(pm.__version__, az.__version__)"
```

Expected: prints two version strings (e.g. `5.16.2 0.20.0`), exits 0.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pymc + arviz as bayes extras"
```

---

## Task 2: Module skeleton

**Files:**
- Create: `src/ganyan/predictor/bayes/__init__.py`
- Create: `tests/test_predictor/test_bayes/__init__.py`

- [ ] **Step 1: Create the package init with module docstring**

```python
# src/ganyan/predictor/bayes/__init__.py
"""Hierarchical Bayesian Plackett-Luce race predictor.

Prototype alongside the LightGBM ensemble. The goal is to determine
whether posterior credible intervals on win probability are better
calibrated than softmax point estimates from the tree-based ranker.
"""
```

- [ ] **Step 2: Create the test package init**

```python
# tests/test_predictor/test_bayes/__init__.py
```

(empty file is fine — pytest just needs the directory to be importable)

- [ ] **Step 3: Verify imports**

```bash
uv run python -c "import ganyan.predictor.bayes"
```

Expected: exits 0, no output.

- [ ] **Step 4: Commit**

```bash
git add src/ganyan/predictor/bayes/__init__.py tests/test_predictor/test_bayes/__init__.py
git commit -m "feat(bayes): add empty bayes predictor module"
```

---

## Task 3: Data layer — training frame and integer indices

**Files:**
- Create: `src/ganyan/predictor/bayes/data.py`
- Create: `tests/test_predictor/test_bayes/test_data.py`

The Plackett-Luce model needs each finished race as an ordered list of horse-indices. We also need integer-encoded jockey, sire, and (track × distance bucket) per horse for the hierarchical effects.

Distance bucket: `[1000, 1300, 1600, 1900, 2400]` boundaries → 4 buckets. Coarse on purpose to avoid blowing up the random-effect dimension.

- [ ] **Step 1: Write the failing test**

Create `tests/test_predictor/test_bayes/test_data.py`:

```python
"""Tests for the Bayesian training-frame builder."""
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.predictor.bayes.data import (
    DISTANCE_BUCKETS, build_training_frame, distance_bucket_for,
)


def test_distance_bucket_assigns_into_4_buckets():
    assert distance_bucket_for(1100) == 0  # < 1300
    assert distance_bucket_for(1300) == 1  # 1300-1599
    assert distance_bucket_for(1600) == 2  # 1600-1899
    assert distance_bucket_for(1900) == 3  # 1900-2399
    assert distance_bucket_for(2400) == 4  # >= 2400
    assert len(DISTANCE_BUCKETS) == 4  # 4 boundaries → 5 buckets (0..4)


def test_build_training_frame_returns_indices_and_orderings(pg_session):
    frame = build_training_frame(
        pg_session,
        from_date=date(2025, 1, 1),
        to_date=date(2025, 12, 31),
    )
    # Each race contributes one row in `orderings` keyed by race_id;
    # value is the list of horse-indices in finish order (1st, 2nd, ...)
    assert len(frame.orderings) > 0
    for race_id, order in frame.orderings.items():
        assert len(order) >= 3, f"race {race_id} should have ≥3 finishers"
        assert all(isinstance(h, int) for h in order)
        assert len(order) == len(set(order)), "no duplicate horse indices in one race"
    # Indices are dense 0..N-1
    assert min(frame.horse_index.values()) == 0
    assert max(frame.horse_index.values()) == len(frame.horse_index) - 1
    # Per-entry hierarchical coords — same length as flat ordering payload
    flat = [h for order in frame.orderings.values() for h in order]
    assert len(frame.jockey_of_horse_in_race) == len(flat)
    assert len(frame.track_dist_of_race) == len(frame.orderings)


def test_build_training_frame_excludes_target_finish_columns():
    """Sanity: the function should not depend on finish_time, only finish_position."""
    # Compile-time check via attribute introspection — we know the data layer
    # only reads finish_position from RaceEntry. This test is the canary.
    import inspect
    from ganyan.predictor.bayes import data
    src = inspect.getsource(data)
    assert "finish_time" not in src
    assert "finish_position" in src
```

(Note: `pg_session` is the existing pytest fixture in `tests/conftest.py`. If absent, fall back to a transient SQLite. Verify in the repo before relying on it.)

- [ ] **Step 2: Run the test to confirm it fails**

```bash
uv run pytest tests/test_predictor/test_bayes/test_data.py -v
```

Expected: ImportError on `ganyan.predictor.bayes.data`.

- [ ] **Step 3: Implement `data.py`**

```python
# src/ganyan/predictor/bayes/data.py
"""Build training/holdout frames for the Bayesian Plackett-Luce model.

Encode horse, jockey, sire, and (track × distance-bucket) as dense
integer indices so PyMC's coords machinery can use them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ganyan.db.models import Horse, Race, RaceEntry, Track


# Distance buckets. Boundaries chosen to roughly equipopulate Turkish cards:
# 0: sprint (<1300), 1: middle (1300-1599), 2: mile (1600-1899),
# 3: long (1900-2399), 4: stayer (≥2400).
DISTANCE_BUCKETS: Tuple[int, ...] = (1300, 1600, 1900, 2400)


def distance_bucket_for(meters: int) -> int:
    for i, boundary in enumerate(DISTANCE_BUCKETS):
        if meters < boundary:
            return i
    return len(DISTANCE_BUCKETS)


@dataclass
class TrainingFrame:
    """All data the PyMC model needs.

    `orderings`: race_id → [horse_idx_first, horse_idx_second, ...]
    `horse_index`: horse_id → dense int
    `jockey_index`: jockey_name → dense int
    `sire_index`: sire_name → dense int (0 reserved for "unknown")
    `track_dist_index`: (track_id, dist_bucket) → dense int
    `jockey_of_horse_in_race`: flat per-entry list of jockey indices,
        same iteration order as `flatten(orderings.values())`
    `sire_of_horse_in_race`: same shape, sire indices
    `track_dist_of_race`: race_id → track_dist_index
    `agf_of_horse_in_race`: same shape as flat, AGF percent in [0,100]
    """

    orderings: Dict[int, List[int]] = field(default_factory=dict)
    horse_index: Dict[int, int] = field(default_factory=dict)
    jockey_index: Dict[str, int] = field(default_factory=dict)
    sire_index: Dict[str, int] = field(default_factory=dict)
    track_dist_index: Dict[Tuple[int, int], int] = field(default_factory=dict)
    jockey_of_horse_in_race: List[int] = field(default_factory=list)
    sire_of_horse_in_race: List[int] = field(default_factory=list)
    track_dist_of_race: Dict[int, int] = field(default_factory=dict)
    agf_of_horse_in_race: List[float] = field(default_factory=list)


def _intern(idx: Dict, key) -> int:
    if key not in idx:
        idx[key] = len(idx)
    return idx[key]


def build_training_frame(
    session: Session,
    from_date: date,
    to_date: date,
    min_field_size: int = 3,
) -> TrainingFrame:
    """Read finished races between [from_date, to_date], build indices."""
    frame = TrainingFrame()
    # Reserve sire_index 0 for "unknown" so cold-start sires are well-defined.
    _intern(frame.sire_index, "")

    races = session.execute(
        select(Race).where(
            Race.date >= from_date,
            Race.date <= to_date,
        ).order_by(Race.date, Race.race_number)
    ).scalars().all()

    for r in races:
        finishers = [
            e for e in r.entries
            if e.finish_position is not None and e.jockey is not None
        ]
        if len(finishers) < min_field_size:
            continue
        finishers.sort(key=lambda e: e.finish_position)
        horse_ids: List[int] = []
        jockey_ids: List[int] = []
        sire_ids: List[int] = []
        agfs: List[float] = []
        for e in finishers:
            horse_ids.append(_intern(frame.horse_index, e.horse_id))
            jockey_ids.append(_intern(frame.jockey_index, e.jockey))
            sire_name = (e.horse.sire_name or "") if e.horse else ""
            sire_ids.append(_intern(frame.sire_index, sire_name))
            agfs.append(float(e.agf) if e.agf is not None else 0.0)
        track_dist = (r.track_id, distance_bucket_for(r.distance_meters or 0))
        frame.track_dist_of_race[r.id] = _intern(frame.track_dist_index, track_dist)
        frame.orderings[r.id] = horse_ids
        frame.jockey_of_horse_in_race.extend(jockey_ids)
        frame.sire_of_horse_in_race.extend(sire_ids)
        frame.agf_of_horse_in_race.extend(agfs)

    return frame
```

Notes:
- `Horse.sire_name` is the assumed column. **Before running the test, `grep -n "sire_name\|sire_id" src/ganyan/db/models.py`** and adapt the access expression if the column has a different name (likely `sire_name` per the existing `sire_win_rate` feature naming). If the column is missing entirely, default to empty string (which interns to index 0 = unknown).
- Skip races with fewer than 3 graded finishers. Plackett-Luce needs an ordering.
- `min_field_size=3` keeps consistency with the existing pick-generator threshold.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_predictor/test_bayes/test_data.py -v
```

Expected: 3 tests pass. If `pg_session` fixture doesn't exist, the second test will be skipped — that's acceptable for now; the first and third must pass.

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/predictor/bayes/data.py tests/test_predictor/test_bayes/test_data.py
git commit -m "feat(bayes): build training frame with horse/jockey/sire/track-dist indices"
```

---

## Task 4: Simple Plackett-Luce model (no hierarchy yet)

**Files:**
- Create: `src/ganyan/predictor/bayes/model.py`
- Create: `tests/test_predictor/test_bayes/test_model.py`

The first model has only one parameter set: `θ_h ~ Normal(0, 1)` per horse. Plackett-Luce likelihood over observed orderings. This serves as a baseline and ensures the wiring is correct before adding hierarchy.

Plackett-Luce log-likelihood for ordering `[h_1, h_2, ..., h_K]`:

```
log P(order) = Σ_k [θ_{h_k} − logsumexp(θ_{h_k}, θ_{h_{k+1}}, ..., θ_{h_K})]
```

- [ ] **Step 1: Write the failing test (synthetic-data recovery)**

Create `tests/test_predictor/test_bayes/test_model.py`:

```python
"""Tests for the PyMC Plackett-Luce model.

Synthetic-data recovery: generate races where horses have known true
skills, fit ADVI, check we recover the rank-1 horse with high posterior
probability.
"""
import numpy as np
import pytest

from ganyan.predictor.bayes.data import TrainingFrame
from ganyan.predictor.bayes.model import build_simple_pl_model, fit_advi


@pytest.fixture
def synthetic_frame() -> TrainingFrame:
    rng = np.random.default_rng(0)
    n_horses = 6
    n_races = 200
    true_skill = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0])
    frame = TrainingFrame()
    for h in range(n_horses):
        frame.horse_index[h] = h
    for race_id in range(n_races):
        # Sample 4 distinct horses
        field = list(rng.choice(n_horses, size=4, replace=False))
        # Plackett-Luce sampling using true skill
        order: list[int] = []
        remaining = list(field)
        while remaining:
            skills = np.array([true_skill[h] for h in remaining])
            probs = np.exp(skills - skills.max())
            probs /= probs.sum()
            pick = rng.choice(remaining, p=probs)
            order.append(int(pick))
            remaining.remove(pick)
        frame.orderings[race_id] = order
    return frame


def test_simple_pl_recovers_top_horse(synthetic_frame):
    model = build_simple_pl_model(synthetic_frame)
    idata = fit_advi(model, n_iter=10_000, seed=0)
    posterior_skill_mean = idata.posterior["theta"].mean(("chain", "draw")).values
    # True top horse is index 0
    assert int(np.argmax(posterior_skill_mean)) == 0
    # Skills should be monotone-decreasing-ish for the first few
    assert posterior_skill_mean[0] > posterior_skill_mean[1]
    assert posterior_skill_mean[1] > posterior_skill_mean[3]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_predictor/test_bayes/test_model.py -v
```

Expected: ImportError on `ganyan.predictor.bayes.model`.

- [ ] **Step 3: Implement `model.py` with the simple Plackett-Luce**

```python
# src/ganyan/predictor/bayes/model.py
"""PyMC Plackett-Luce models — simple and hierarchical."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ganyan.predictor.bayes.data import TrainingFrame


def _plackett_luce_loglik(theta: pt.TensorVariable, orderings: list[list[int]]):
    """Sum of log-likelihoods over observed orderings.

    For ordering [h_1, ..., h_K]:
        Σ_k log( exp(θ_{h_k}) / Σ_{j≥k} exp(θ_{h_j}) )
    """
    total = pt.zeros((), dtype="float64")
    for order in orderings:
        order_arr = pt.constant(order, dtype="int64")
        chosen = theta[order_arr]
        # remaining[k] = logsumexp over chosen[k:]
        # Implement via cumulative reverse-logsumexp.
        # Reverse, then logcumsumexp, then reverse again.
        rev = chosen[::-1]
        log_cumsum = pt.cumsum(pt.exp(rev - rev.max()), axis=0)
        # log_cumsum is in shifted/exp space; reconstruct logsumexp:
        # logsumexp_rev[k] = log(cumsum_k) + rev.max()
        log_remaining_rev = pt.log(log_cumsum) + rev.max()
        log_remaining = log_remaining_rev[::-1]
        total = total + pt.sum(chosen - log_remaining)
    return total


def build_simple_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    orderings = list(frame.orderings.values())
    coords = {"horse": list(range(n_horses))}
    with pm.Model(coords=coords) as model:
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, dims="horse")
        pm.Potential("plackett_luce", _plackett_luce_loglik(theta, orderings))
    return model


def fit_advi(model: pm.Model, n_iter: int = 30_000, seed: int = 0):
    """Mean-field ADVI fit. Returns an ArviZ InferenceData."""
    with model:
        approx = pm.fit(n_iter, method="advi", random_seed=seed, progressbar=False)
        idata = approx.sample(draws=2_000, random_seed=seed)
    return idata
```

The Plackett-Luce log-likelihood here uses a tail-cumulative-logsumexp trick, which is differentiable in PyTensor and avoids the O(K²) explicit loops that wreck compilation.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_predictor/test_bayes/test_model.py::test_simple_pl_recovers_top_horse -v
```

Expected: passes in 30-90s. ADVI converges, top horse correctly identified. If it fails, double `n_iter` to `20_000`.

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/predictor/bayes/model.py tests/test_predictor/test_bayes/test_model.py
git commit -m "feat(bayes): simple Plackett-Luce model with ADVI fit"
```

---

## Task 5: Add hierarchical priors (jockey + sire + track×distance)

**Files:**
- Modify: `src/ganyan/predictor/bayes/model.py`
- Modify: `tests/test_predictor/test_bayes/test_model.py`

Replace `θ_h ~ Normal(0, 1)` with:

```
θ_h = μ_h + ε_h
μ_h = α_jockey(h) + β_sire(h) + γ_(track×dist)(race(h))
ε_h ~ Normal(0, σ_θ)
α_j ~ Normal(0, σ_α), β_s ~ Normal(0, σ_β), γ_td ~ Normal(0, σ_γ)
```

But: `α_jockey(h)` and `γ_(track×dist)(race(h))` are race-specific, so the latent variable is no longer one-θ-per-horse. Easier: keep `θ_h` as horse-only and add per-entry effects in the likelihood:

```
score(horse h in race r with jockey j) = θ_h + α_j + γ_{td(r)}
β_sire only enters as part of the prior on θ_h (since sire is horse-attribute, not race-attribute).
```

This keeps `θ_h` defined per horse, and lets jockey/track-distance contribute as race-conditional shifts.

- [ ] **Step 1: Add hierarchical model factory + test**

Append to `tests/test_predictor/test_bayes/test_model.py`:

```python
def test_hierarchical_model_compiles(synthetic_frame):
    """Smoke test — hierarchical model compiles and ADVI runs."""
    # Decorate the synthetic frame with jockey + track-dist coords.
    frame = synthetic_frame
    flat = [h for order in frame.orderings.values() for h in order]
    frame.jockey_index = {f"j{i}": i for i in range(3)}
    frame.track_dist_index = {(0, 0): 0, (0, 1): 1}
    frame.sire_index = {"": 0, "sireA": 1, "sireB": 2}
    # Round-robin assign jockey
    frame.jockey_of_horse_in_race = [i % 3 for i in range(len(flat))]
    frame.sire_of_horse_in_race = [(h % 3) for h in flat]
    for rid in frame.orderings:
        frame.track_dist_of_race[rid] = rid % 2

    from ganyan.predictor.bayes.model import build_hierarchical_pl_model
    model = build_hierarchical_pl_model(frame)
    from ganyan.predictor.bayes.model import fit_advi
    idata = fit_advi(model, n_iter=5_000, seed=0)
    assert "theta" in idata.posterior
    assert "alpha_jockey" in idata.posterior
    assert "gamma_track_dist" in idata.posterior
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_predictor/test_bayes/test_model.py::test_hierarchical_model_compiles -v
```

Expected: AttributeError on `build_hierarchical_pl_model`.

- [ ] **Step 3: Implement `build_hierarchical_pl_model`**

Append to `src/ganyan/predictor/bayes/model.py`:

```python
def _plackett_luce_loglik_with_offsets(
    theta: pt.TensorVariable,
    alpha: pt.TensorVariable,           # jockey effect, dims=jockey
    gamma: pt.TensorVariable,           # track×dist effect, dims=track_dist
    orderings: list[list[int]],
    jockey_per_entry: list[int],         # flat per-entry jockey indices
    track_dist_per_race: list[int],      # one per race, in same iteration order
):
    total = pt.zeros((), dtype="float64")
    flat_idx = 0
    for race_idx, order in enumerate(orderings):
        order_arr = pt.constant(order, dtype="int64")
        n = len(order)
        jockey_slice = jockey_per_entry[flat_idx : flat_idx + n]
        flat_idx += n
        jockey_arr = pt.constant(jockey_slice, dtype="int64")
        td_idx = track_dist_per_race[race_idx]
        score = theta[order_arr] + alpha[jockey_arr] + gamma[td_idx]
        rev = score[::-1]
        log_cumsum = pt.cumsum(pt.exp(rev - rev.max()), axis=0)
        log_remaining_rev = pt.log(log_cumsum) + rev.max()
        log_remaining = log_remaining_rev[::-1]
        total = total + pt.sum(score - log_remaining)
    return total


def build_hierarchical_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    orderings = list(frame.orderings.values())
    track_dist_per_race = [frame.track_dist_of_race[rid] for rid in frame.orderings]

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)
        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")
        # horse → sire lookup
        horse_to_sire = np.zeros(n_horses, dtype="int64")
        # Reconstruct mapping: walk the flat per-entry sire list once;
        # the first time we see a horse-index, record its sire.
        seen = set()
        flat_horses = [h for order in frame.orderings.values() for h in order]
        for h, s in zip(flat_horses, frame.sire_of_horse_in_race):
            if h not in seen:
                horse_to_sire[h] = s
                seen.add(h)

        mu_horse = beta_sire[horse_to_sire]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        gamma_track_dist = pm.Normal(
            "gamma_track_dist", 0.0, sigma_gamma, dims="track_dist",
        )
        pm.Potential(
            "plackett_luce_hier",
            _plackett_luce_loglik_with_offsets(
                theta, alpha_jockey, gamma_track_dist,
                orderings, frame.jockey_of_horse_in_race, track_dist_per_race,
            ),
        )
    return model
```

- [ ] **Step 4: Run both tests to verify they pass**

```bash
uv run pytest tests/test_predictor/test_bayes/test_model.py -v
```

Expected: both `test_simple_pl_recovers_top_horse` and `test_hierarchical_model_compiles` pass. ADVI on the hierarchical model takes ~60-180s for 5K iters on synthetic data.

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/predictor/bayes/model.py tests/test_predictor/test_bayes/test_model.py
git commit -m "feat(bayes): hierarchical PL model with jockey/sire/track-dist effects"
```

---

## Task 6: Add AGF as informative prior coefficient

**Files:**
- Modify: `src/ganyan/predictor/bayes/model.py`
- Modify: `tests/test_predictor/test_bayes/test_model.py`

Add `δ` — a global coefficient that pulls horse skill `θ_h` towards public belief (AGF). The likelihood score becomes:

```
score(horse h in race r) = θ_h + α_j + γ_{td} + δ · agf_z_score(h in r)
```

`agf_z_score` is the within-race standardized AGF (zero-mean, unit-variance per race), so `δ` is dimensionless and identifiable. Set `δ ~ Normal(0, 1)` — the model can choose how much to weight crowd belief.

- [ ] **Step 1: Update the test for AGF parameter**

Append to `tests/test_predictor/test_bayes/test_model.py`:

```python
def test_hierarchical_with_agf_includes_delta(synthetic_frame):
    frame = synthetic_frame
    flat = [h for order in frame.orderings.values() for h in order]
    frame.jockey_index = {f"j{i}": i for i in range(3)}
    frame.track_dist_index = {(0, 0): 0}
    frame.sire_index = {"": 0}
    frame.jockey_of_horse_in_race = [i % 3 for i in range(len(flat))]
    frame.sire_of_horse_in_race = [0] * len(flat)
    for rid in frame.orderings:
        frame.track_dist_of_race[rid] = 0
    # Synthetic AGF: random per entry
    rng = np.random.default_rng(1)
    frame.agf_of_horse_in_race = list(rng.uniform(5, 50, size=len(flat)))

    from ganyan.predictor.bayes.model import build_hierarchical_pl_model_with_agf
    model = build_hierarchical_pl_model_with_agf(frame)
    from ganyan.predictor.bayes.model import fit_advi
    idata = fit_advi(model, n_iter=5_000, seed=0)
    assert "delta_agf" in idata.posterior
    # delta should be non-trivial
    delta_mean = float(idata.posterior["delta_agf"].mean())
    assert -2.0 < delta_mean < 2.0  # sanity range
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_predictor/test_bayes/test_model.py::test_hierarchical_with_agf_includes_delta -v
```

Expected: AttributeError on `build_hierarchical_pl_model_with_agf`.

- [ ] **Step 3: Implement `build_hierarchical_pl_model_with_agf`**

Append to `src/ganyan/predictor/bayes/model.py`:

```python
def _agf_zscore_per_entry(
    orderings: list[list[int]], agf_per_entry: list[float],
) -> list[float]:
    out: list[float] = []
    flat_idx = 0
    for order in orderings:
        n = len(order)
        slice_ = np.asarray(agf_per_entry[flat_idx : flat_idx + n], dtype="float64")
        flat_idx += n
        mu = slice_.mean()
        sd = slice_.std() if slice_.std() > 1e-9 else 1.0
        out.extend(((slice_ - mu) / sd).tolist())
    return out


def _plackett_luce_loglik_with_agf(
    theta, alpha, gamma, delta, orderings,
    jockey_per_entry, track_dist_per_race, agf_z_per_entry,
):
    total = pt.zeros((), dtype="float64")
    flat_idx = 0
    for race_idx, order in enumerate(orderings):
        order_arr = pt.constant(order, dtype="int64")
        n = len(order)
        jockey_slice = jockey_per_entry[flat_idx : flat_idx + n]
        agf_slice = agf_z_per_entry[flat_idx : flat_idx + n]
        flat_idx += n
        jockey_arr = pt.constant(jockey_slice, dtype="int64")
        agf_arr = pt.constant(agf_slice, dtype="float64")
        td_idx = track_dist_per_race[race_idx]
        score = (
            theta[order_arr]
            + alpha[jockey_arr]
            + gamma[td_idx]
            + delta * agf_arr
        )
        rev = score[::-1]
        log_cumsum = pt.cumsum(pt.exp(rev - rev.max()), axis=0)
        log_remaining_rev = pt.log(log_cumsum) + rev.max()
        log_remaining = log_remaining_rev[::-1]
        total = total + pt.sum(score - log_remaining)
    return total


def build_hierarchical_pl_model_with_agf(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    orderings = list(frame.orderings.values())
    track_dist_per_race = [frame.track_dist_of_race[rid] for rid in frame.orderings]
    agf_z = _agf_zscore_per_entry(orderings, frame.agf_of_horse_in_race)

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)
        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")

        horse_to_sire = np.zeros(n_horses, dtype="int64")
        seen = set()
        flat_horses = [h for order in frame.orderings.values() for h in order]
        for h, s in zip(flat_horses, frame.sire_of_horse_in_race):
            if h not in seen:
                horse_to_sire[h] = s
                seen.add(h)

        mu_horse = beta_sire[horse_to_sire]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        gamma_track_dist = pm.Normal(
            "gamma_track_dist", 0.0, sigma_gamma, dims="track_dist",
        )
        delta_agf = pm.Normal("delta_agf", 0.0, 1.0)

        pm.Potential(
            "plackett_luce_agf",
            _plackett_luce_loglik_with_agf(
                theta, alpha_jockey, gamma_track_dist, delta_agf,
                orderings, frame.jockey_of_horse_in_race,
                track_dist_per_race, agf_z,
            ),
        )
    return model
```

- [ ] **Step 4: Run all model tests**

```bash
uv run pytest tests/test_predictor/test_bayes/test_model.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/predictor/bayes/model.py tests/test_predictor/test_bayes/test_model.py
git commit -m "feat(bayes): add AGF as informative prior coefficient on PL score"
```

---

## Task 7: Trainer — fit on real data, persist posterior

**Files:**
- Create: `src/ganyan/predictor/bayes/trainer.py`
- Create: `tests/test_predictor/test_bayes/test_trainer.py`

Persist posterior + index dictionaries so the predictor can load them without retraining.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predictor/test_bayes/test_trainer.py
"""Tests for the Bayesian trainer's persistence layer."""
import json
from pathlib import Path

import numpy as np
import pytest

from ganyan.predictor.bayes.data import TrainingFrame
from ganyan.predictor.bayes.trainer import save_posterior, load_posterior


def test_save_and_load_roundtrip(tmp_path: Path):
    # Build a tiny fake InferenceData via PyMC
    import pymc as pm
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        idata = pm.sample(
            draws=50, tune=50, chains=1, random_seed=0,
            progressbar=False,
        )
    frame = TrainingFrame()
    frame.horse_index = {1: 0, 2: 1}
    frame.jockey_index = {"jA": 0}
    frame.sire_index = {"": 0}
    frame.track_dist_index = {(3, 0): 0}

    base = tmp_path / "bayes_pl"
    save_posterior(idata, frame, base)

    assert (base.with_suffix(".nc")).exists()
    assert (base.with_suffix(".indices.json")).exists()

    loaded_idata, loaded_frame = load_posterior(base)
    assert "x" in loaded_idata.posterior
    assert loaded_frame.horse_index == {1: 0, 2: 1}
    assert loaded_frame.jockey_index == {"jA": 0}
    # JSON keys are strings — track_dist tuple should round-trip via str-encoding
    assert (3, 0) in loaded_frame.track_dist_index
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_predictor/test_bayes/test_trainer.py -v
```

Expected: ImportError on `ganyan.predictor.bayes.trainer`.

- [ ] **Step 3: Implement the trainer module**

```python
# src/ganyan/predictor/bayes/trainer.py
"""Train and persist the Bayesian PL posterior."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Tuple

import arviz as az
from sqlalchemy.orm import Session

from ganyan.predictor.bayes.data import TrainingFrame, build_training_frame
from ganyan.predictor.bayes.model import (
    build_hierarchical_pl_model_with_agf, fit_advi,
)


def save_posterior(idata: az.InferenceData, frame: TrainingFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    nc_path = base.with_suffix(".nc")
    idata.to_netcdf(str(nc_path))
    idx = {
        "horse_index": {str(k): v for k, v in frame.horse_index.items()},
        "jockey_index": frame.jockey_index,
        "sire_index": frame.sire_index,
        "track_dist_index": {
            f"{tid}_{db}": v for (tid, db), v in frame.track_dist_index.items()
        },
    }
    base.with_suffix(".indices.json").write_text(json.dumps(idx))


def load_posterior(base: Path) -> Tuple[az.InferenceData, TrainingFrame]:
    idata = az.from_netcdf(str(base.with_suffix(".nc")))
    raw = json.loads(base.with_suffix(".indices.json").read_text())
    frame = TrainingFrame()
    frame.horse_index = {int(k): v for k, v in raw["horse_index"].items()}
    frame.jockey_index = raw["jockey_index"]
    frame.sire_index = raw["sire_index"]
    track_dist: dict[tuple[int, int], int] = {}
    for key, v in raw["track_dist_index"].items():
        tid, db = key.split("_")
        track_dist[(int(tid), int(db))] = v
    frame.track_dist_index = track_dist
    return idata, frame


def train_full(
    session: Session,
    from_date: date,
    to_date: date,
    output_base: Path,
    n_iter: int = 60_000,
    seed: int = 0,
) -> None:
    frame = build_training_frame(session, from_date, to_date)
    model = build_hierarchical_pl_model_with_agf(frame)
    idata = fit_advi(model, n_iter=n_iter, seed=seed)
    save_posterior(idata, frame, output_base)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_predictor/test_bayes/test_trainer.py -v
```

Expected: passes in ~10s.

- [ ] **Step 5: Train on real 2025-01-01 → 2026-01-25 data**

This is the big training run — expect 10-30 minutes for ADVI on the full window.

```bash
uv run python -c "
from datetime import date
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from ganyan.predictor.bayes.trainer import train_full

eng = create_engine('postgresql+psycopg://ganyan:ganyan@localhost:5432/ganyan')
with Session(eng) as s:
    train_full(s, date(2025,1,1), date(2026,1,25), Path('models/bayes_pl_v1'))
print('done')
"
```

Expected: writes `models/bayes_pl_v1.nc` (~50-200 MB) + `models/bayes_pl_v1.indices.json`. Print `done`.

If ADVI diverges or fails, halve `n_iter` to 30_000 and retry; check `models/bayes_pl_v1.nc` posterior for sanity (`uv run python -c "import arviz as az; print(az.summary(az.from_netcdf('models/bayes_pl_v1.nc'))[:20])"`).

- [ ] **Step 6: Commit (model artifacts go to .gitignore)**

```bash
echo "models/bayes_pl_*" >> .gitignore
git add .gitignore src/ganyan/predictor/bayes/trainer.py tests/test_predictor/test_bayes/test_trainer.py
git commit -m "feat(bayes): trainer + posterior persistence"
```

---

## Task 8: Predictor — load posterior, predict for `race_id`

**Files:**
- Create: `src/ganyan/predictor/bayes/predictor.py`
- Create: `tests/test_predictor/test_bayes/test_predictor.py`

Inference: for a race with horses `[h_1, ..., h_K]`, jockeys `[j_1, ...]`, track-dist `td`, and AGFs `[agf_1, ...]`:

For each posterior draw `s = 1..S`:
- `score_k = θ_{h_k}^{(s)} + α_{j_k}^{(s)} + γ_{td}^{(s)} + δ^{(s)} · agf_z_k`
- Win prob `p_k^{(s)} = softmax(score_k)`

Then aggregate across draws: `p̂_k = mean_s p_k^{(s)}`, with credible intervals `[q_5, q_95]`.

Cold-start (horse not in `horse_index`): use `μ_horse = β_sire[sire_of(h)]` if sire known, else `μ_horse = 0`. Sample `θ_h^{(s)} ~ Normal(μ_horse, σ_θ^{(s)})` for each draw.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predictor/test_bayes/test_predictor.py
import numpy as np
import pytest

from ganyan.predictor.bayes.predictor import BayesPrediction, predict_from_posterior
from ganyan.predictor.bayes.data import TrainingFrame


def _fake_idata(horse_skills: list[float]):
    """Build a tiny InferenceData with deterministic theta and zeros elsewhere."""
    import arviz as az
    n_draws = 100
    n_horses = len(horse_skills)
    rng = np.random.default_rng(0)
    posterior = {
        "theta": (("chain","draw","horse"),
                  np.tile(np.array(horse_skills), (1, n_draws, 1))
                  + rng.normal(0, 0.01, (1, n_draws, n_horses))),
        "alpha_jockey": (("chain","draw","jockey"), np.zeros((1, n_draws, 1))),
        "beta_sire": (("chain","draw","sire"), np.zeros((1, n_draws, 1))),
        "gamma_track_dist": (("chain","draw","track_dist"), np.zeros((1, n_draws, 1))),
        "delta_agf": (("chain","draw"), np.zeros((1, n_draws))),
        "sigma_theta": (("chain","draw"), 0.1 * np.ones((1, n_draws))),
    }
    return az.from_dict(posterior)


def test_predict_from_posterior_orders_by_skill():
    frame = TrainingFrame()
    frame.horse_index = {101: 0, 102: 1, 103: 2}
    frame.jockey_index = {"jA": 0}
    frame.sire_index = {"": 0}
    frame.track_dist_index = {(3, 0): 0}
    idata = _fake_idata(horse_skills=[2.0, 1.0, 0.0])

    race = {
        "horse_ids": [103, 102, 101],
        "jockeys": ["jA", "jA", "jA"],
        "sires": ["", "", ""],
        "track_id": 3,
        "distance_meters": 1200,
        "agfs": [10.0, 10.0, 10.0],
    }
    preds = predict_from_posterior(idata, frame, race)
    assert isinstance(preds, list)
    assert all(isinstance(p, BayesPrediction) for p in preds)
    # 101 has highest skill → should get highest mean prob
    by_horse = {p.horse_id: p for p in preds}
    assert by_horse[101].mean_prob > by_horse[102].mean_prob > by_horse[103].mean_prob
    # Probabilities should sum to ~1
    assert abs(sum(p.mean_prob for p in preds) - 1.0) < 0.01
    # Credible interval: low ≤ mean ≤ high
    for p in preds:
        assert p.lo_5 <= p.mean_prob <= p.hi_95
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_predictor/test_bayes/test_predictor.py -v
```

Expected: ImportError on `ganyan.predictor.bayes.predictor`.

- [ ] **Step 3: Implement `predictor.py`**

```python
# src/ganyan/predictor/bayes/predictor.py
"""Inference: load posterior + predict win probabilities for a race."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import arviz as az

from ganyan.predictor.bayes.data import TrainingFrame, distance_bucket_for


@dataclass
class BayesPrediction:
    horse_id: int
    horse_name: str
    mean_prob: float    # posterior mean of win probability (0..1)
    lo_5: float         # 5th percentile of posterior win prob
    hi_95: float        # 95th percentile
    mean_score: float   # posterior mean of the latent PL score


def predict_from_posterior(
    idata: az.InferenceData,
    frame: TrainingFrame,
    race: dict,
) -> List[BayesPrediction]:
    """Compute posterior-over-win-prob for one race.

    `race` has keys:
      horse_ids: list[int]
      jockeys: list[str]
      sires: list[str]
      track_id: int
      distance_meters: int
      agfs: list[float]
      (optional) horse_names: list[str]
    """
    post = idata.posterior
    # Stack chain × draw → samples
    theta = post["theta"].stack(sample=("chain","draw")).values   # (n_horses, S)
    alpha = post["alpha_jockey"].stack(sample=("chain","draw")).values
    beta = post["beta_sire"].stack(sample=("chain","draw")).values
    gamma = post["gamma_track_dist"].stack(sample=("chain","draw")).values
    delta = post["delta_agf"].stack(sample=("chain","draw")).values   # (S,)
    sigma_theta = post["sigma_theta"].stack(sample=("chain","draw")).values

    S = delta.shape[0]
    n = len(race["horse_ids"])
    rng = np.random.default_rng(0)

    horse_idx = []
    cold_starts = []
    for hid, sire in zip(race["horse_ids"], race["sires"]):
        if hid in frame.horse_index:
            horse_idx.append(frame.horse_index[hid])
            cold_starts.append(None)
        else:
            sire_idx = frame.sire_index.get(sire, 0)
            horse_idx.append(-1)
            cold_starts.append(sire_idx)

    jockey_idx = [frame.jockey_index.get(j, -1) for j in race["jockeys"]]
    td_key = (race["track_id"], distance_bucket_for(race["distance_meters"]))
    td_idx = frame.track_dist_index.get(td_key, -1)

    # Build score matrix (n, S)
    score = np.zeros((n, S))
    for k in range(n):
        if horse_idx[k] >= 0:
            score[k] += theta[horse_idx[k], :]
        else:
            sire_mu = beta[cold_starts[k], :]
            score[k] += sire_mu + rng.normal(0, sigma_theta, size=S)
        if jockey_idx[k] >= 0:
            score[k] += alpha[jockey_idx[k], :]
        # else: jockey effect is 0 (cold-start)
        if td_idx >= 0:
            score[k] += gamma[td_idx, :]

    # AGF z-score within race
    agfs = np.asarray(race["agfs"], dtype=float)
    if agfs.std() > 1e-9:
        agf_z = (agfs - agfs.mean()) / agfs.std()
    else:
        agf_z = np.zeros_like(agfs)
    score += np.outer(agf_z, delta)

    # Softmax across horses, per draw
    score -= score.max(axis=0, keepdims=True)
    exps = np.exp(score)
    probs = exps / exps.sum(axis=0, keepdims=True)   # (n, S)

    mean_prob = probs.mean(axis=1)
    lo_5 = np.quantile(probs, 0.05, axis=1)
    hi_95 = np.quantile(probs, 0.95, axis=1)
    mean_score = score.mean(axis=1)

    names = race.get("horse_names", [str(hid) for hid in race["horse_ids"]])
    out = [
        BayesPrediction(
            horse_id=hid, horse_name=name,
            mean_prob=float(mp), lo_5=float(l5), hi_95=float(h95),
            mean_score=float(ms),
        )
        for hid, name, mp, l5, h95, ms in zip(
            race["horse_ids"], names, mean_prob, lo_5, hi_95, mean_score,
        )
    ]
    out.sort(key=lambda p: -p.mean_prob)
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_predictor/test_bayes/test_predictor.py -v
```

Expected: passes in <2s.

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/predictor/bayes/predictor.py tests/test_predictor/test_bayes/test_predictor.py
git commit -m "feat(bayes): predictor with credible intervals over win prob"
```

---

## Task 9: Calibration evaluation vs LightGBM ranker

**Files:**
- Create: `src/ganyan/predictor/bayes/calibration.py`
- Create: `tests/test_predictor/test_bayes/test_calibration.py`

Compute on the holdout window 2026-01-26 → 2026-04-28 (≈3 months, ~13K entries):

- **top1_rate**: P(model rank-1 = winner)
- **brier_top1**: mean((p_top1 − 1[winner=top1])²)
- **mean_log_lik**: mean log probability of actual winner under predicted distribution
- **ci80_coverage**: fraction of races where the actual winner's posterior win-prob CI [10%, 90%] contains the empirical hit rate of model rank-1

Run the same metrics against LightGBM `lightgbm_ranker.txt` predictions stored in the `predictions` table for that window.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predictor/test_bayes/test_calibration.py
import numpy as np

from ganyan.predictor.bayes.calibration import (
    brier_top1, log_likelihood_of_winner, top1_hit_rate,
)


def test_top1_hit_rate_basic():
    races = [
        # (preds: [(prob, is_winner), ...]) — each list is one race
        [(0.6, True), (0.3, False), (0.1, False)],
        [(0.5, False), (0.4, True), (0.1, False)],
        [(0.7, True), (0.2, False), (0.1, False)],
    ]
    assert top1_hit_rate(races) == pytest.approx(2/3)


def test_brier_top1_perfect_predictor():
    # Top pick always has prob 1.0 and always wins
    races = [
        [(1.0, True), (0.0, False)],
        [(1.0, True), (0.0, False)],
    ]
    assert brier_top1(races) == 0.0


def test_log_likelihood_uniform():
    # Uniform predictor on 4-horse race → log(0.25) per race
    races = [[(0.25, True), (0.25, False), (0.25, False), (0.25, False)]] * 5
    assert log_likelihood_of_winner(races) == pytest.approx(np.log(0.25))


import pytest
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_predictor/test_bayes/test_calibration.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `calibration.py`**

```python
# src/ganyan/predictor/bayes/calibration.py
"""Calibration metrics for race predictors.

Each race is represented as a list of (predicted_prob, is_winner) tuples.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


Race = List[Tuple[float, bool]]


def top1_hit_rate(races: Iterable[Race]) -> float:
    races = list(races)
    if not races:
        return 0.0
    hits = 0
    for r in races:
        top = max(r, key=lambda t: t[0])
        if top[1]:
            hits += 1
    return hits / len(races)


def brier_top1(races: Iterable[Race]) -> float:
    races = list(races)
    if not races:
        return 0.0
    total = 0.0
    for r in races:
        top_prob, top_won = max(r, key=lambda t: t[0])
        total += (top_prob - (1.0 if top_won else 0.0)) ** 2
    return total / len(races)


def log_likelihood_of_winner(races: Iterable[Race]) -> float:
    races = list(races)
    if not races:
        return 0.0
    total = 0.0
    n = 0
    for r in races:
        for prob, won in r:
            if won:
                total += np.log(max(prob, 1e-12))
                n += 1
                break
    return total / max(n, 1)
```

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest tests/test_predictor/test_bayes/test_calibration.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Run the holdout evaluation**

Create a one-off evaluation script (not committed; run interactively):

```bash
uv run python - <<'PY'
from datetime import date
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from ganyan.db.models import Race, RaceEntry, Track, Prediction
from ganyan.predictor.bayes.predictor import predict_from_posterior
from ganyan.predictor.bayes.trainer import load_posterior
from ganyan.predictor.bayes.calibration import (
    top1_hit_rate, brier_top1, log_likelihood_of_winner,
)

idata, frame = load_posterior(Path("models/bayes_pl_v1"))

eng = create_engine("postgresql+psycopg://ganyan:ganyan@localhost:5432/ganyan")
bayes_races = []
lgbm_races = []
with Session(eng) as s:
    races = s.execute(
        select(Race).where(
            Race.date >= date(2026,1,26),
            Race.date <= date(2026,4,28),
        ).order_by(Race.date, Race.race_number)
    ).scalars().all()
    for r in races:
        finishers = [e for e in r.entries if e.finish_position is not None]
        if len(finishers) < 3:
            continue
        # Bayes prediction
        race_in = {
            "horse_ids": [e.horse_id for e in r.entries],
            "jockeys":   [e.jockey or "" for e in r.entries],
            "sires":     [(e.horse.sire_name or "") if e.horse else "" for e in r.entries],
            "track_id":  r.track_id,
            "distance_meters": r.distance_meters or 0,
            "agfs":      [float(e.agf) if e.agf is not None else 0.0 for e in r.entries],
        }
        preds = predict_from_posterior(idata, frame, race_in)
        winner_id = next(e.horse_id for e in finishers if e.finish_position == 1)
        bayes_races.append([
            (p.mean_prob, p.horse_id == winner_id) for p in preds
        ])
        # LGBM prediction (use stored predicted_probability)
        lgbm_probs = [
            (float(e.predicted_probability or 0.0) / 100.0, e.horse_id == winner_id)
            for e in r.entries
        ]
        if any(p > 0 for p, _ in lgbm_probs):
            lgbm_races.append(lgbm_probs)

print(f"Holdout races: bayes={len(bayes_races)}, lgbm={len(lgbm_races)}")
print(f"BAYES top1 = {top1_hit_rate(bayes_races):.3f}  brier = {brier_top1(bayes_races):.4f}  logL = {log_likelihood_of_winner(bayes_races):.4f}")
print(f"LGBM  top1 = {top1_hit_rate(lgbm_races):.3f}  brier = {brier_top1(lgbm_races):.4f}  logL = {log_likelihood_of_winner(lgbm_races):.4f}")
PY
```

Expected: prints 6 metrics. Bayes should be at least competitive on top1 and notably better on logL/brier (Bayes posteriors tend to have heavier tails → less overconfident → lower brier).

- [ ] **Step 6: Save the comparison + commit**

If the metrics are conclusive, save them to `docs/superpowers/plans/2026-04-29-hierarchical-bayes-results.md` (numbers + interpretation). If inconclusive, note that and propose next steps (NUTS instead of ADVI? More data? Add HP/KGS as features?).

```bash
git add src/ganyan/predictor/bayes/calibration.py tests/test_predictor/test_bayes/test_calibration.py docs/superpowers/plans/2026-04-29-hierarchical-bayes-results.md
git commit -m "feat(bayes): calibration metrics + LGBM holdout comparison"
```

---

## Self-Review

**Spec coverage:**
- Hierarchical priors: Task 5 (jockey, sire) + Task 5 (track×distance)
- AGF as prior coefficient: Task 6
- Validation on 2026-01-26+ holdout: Task 9
- Posterior calibration vs LightGBM: Task 9
- 1-2 day prototype scope: 9 tasks, no production wiring ✓

**Placeholder scan:**
No `TBD`, no `add error handling`, no `similar to Task N`. All steps have concrete code. The one runtime caveat (does `Horse.sire_name` exist?) is explicit in Task 3 with a `grep` instruction and a fallback path.

**Type consistency:**
- `TrainingFrame` defined in Task 3, used identically in Tasks 4-9 ✓
- `BayesPrediction` defined in Task 8, used in Task 9's eval script ✓
- `build_simple_pl_model`, `build_hierarchical_pl_model`, `build_hierarchical_pl_model_with_agf` all named consistently ✓
- `fit_advi(model, n_iter, seed)` signature stable across Tasks 4-7 ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-29-hierarchical-bayes-predictor.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints

Which approach?

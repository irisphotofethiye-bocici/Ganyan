"""LightGBM LambdaRank trainer with temporal holdout."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ganyan.predictor.ml.features import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    TrainingFrame,
    build_training_frame,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3].parent / "models"
DEFAULT_MODEL_BASENAME = "lightgbm_ranker"

_DEFAULT_LGBM_PARAMS: dict = {
    "objective": "lambdarank",
    # Evaluate on a richer range of cut-offs than just top-1.  NDCG@1
    # saturates after a single good tree when one feature (AGF) already
    # sorts the leader correctly most of the time; @5/@10 expose
    # residual ranking improvements the other features can still add.
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "verbose": -1,
    "lambdarank_truncation_level": 10,
}

# Default patience: generous so lambdarank has room to keep refining
# when the top-ranker metric plateaus.
_DEFAULT_EARLY_STOPPING_ROUNDS = 100


@dataclass
class TrainingResult:
    """What the trainer produces.

    Attributes
    ----------
    model_path:
        Absolute path to the saved LightGBM model file.
    metadata_path:
        Path to the JSON sidecar with feature list / training metadata.
    train_races, test_races:
        Race counts in each split.
    metrics:
        Holdout metrics — top-1 accuracy, top-3 accuracy, mean winner
        rank, NDCG@{1,3}.
    feature_importance:
        Feature → importance (gain) dict sorted descending.
    """

    model_path: Path
    metadata_path: Path
    train_races: int
    test_races: int
    metrics: dict[str, float]
    feature_importance: dict[str, float] = field(default_factory=dict)


def _temporal_split(
    frame: TrainingFrame, holdout_fraction: float,
) -> tuple[TrainingFrame, TrainingFrame]:
    """Split a training frame at the date quantile."""
    unique_dates = sorted(frame.race_dates.unique())
    if len(unique_dates) < 2:
        # Too little data to hold anything out.
        return frame, TrainingFrame(
            features=frame.features.iloc[:0].copy(),
            target=frame.target.iloc[:0].copy(),
            groups=frame.groups.iloc[:0].copy(),
            race_dates=frame.race_dates.iloc[:0].copy(),
        )

    cutoff_idx = max(1, int(len(unique_dates) * (1.0 - holdout_fraction)))
    cutoff_date = unique_dates[cutoff_idx]

    train_mask = frame.race_dates < cutoff_date
    test_mask = ~train_mask

    def _slice(mask: pd.Series) -> TrainingFrame:
        return TrainingFrame(
            features=frame.features[mask].reset_index(drop=True),
            target=frame.target[mask].reset_index(drop=True),
            groups=frame.groups[mask].reset_index(drop=True),
            race_dates=frame.race_dates[mask].reset_index(drop=True),
        )

    return _slice(train_mask), _slice(test_mask)


def _fit_temperature(
    model: lgb.Booster, frame: TrainingFrame,
) -> float:
    """Fit a single scalar ``T`` minimising NLL of the within-race softmax.

    LambdaRank margins have an arbitrary scale — one booster might output
    scores in ``[-2, 4]`` and another in ``[-0.3, 0.6]`` for the same
    underlying belief.  A raw softmax on those scales is wildly over- or
    under-confident, and Kelly sizing downstream cares a great deal about
    calibration.  Fitting one scalar temperature on the holdout
    (Guo et al., 2017, "On Calibration of Modern Neural Networks" —
    simple Platt scaling for multi-class) corrects the aggregate
    confidence without touching the ranking.

    Optimises NLL over ``T ∈ [0.05, 20]`` via golden-section search on
    ``log T`` — one scalar, convex objective, 40 evaluations is enough
    for 1e-4 precision.  Returns ``1.0`` if holdout is empty or has no
    identifiable winner per race.
    """
    if frame.features.empty:
        return 1.0

    raw = model.predict(frame.features)
    df = frame.features.copy()
    df["_score"] = raw
    df["_target"] = frame.target.values
    df["_race"] = frame.groups.values

    race_blocks: list[tuple[np.ndarray, int]] = []
    for _race_id, race_df in df.groupby("_race", sort=False):
        scores = race_df["_score"].to_numpy(dtype=float)
        targets = race_df["_target"].to_numpy()
        winner_target = targets.max()
        winner_rows = np.where(targets == winner_target)[0]
        if winner_rows.size == 0:
            continue
        race_blocks.append((scores, int(winner_rows[0])))
    if not race_blocks:
        return 1.0

    def nll(t: float) -> float:
        total = 0.0
        for scores, widx in race_blocks:
            shifted = scores / t
            shifted -= shifted.max()
            exps = np.exp(shifted)
            p_w = exps[widx] / exps.sum()
            # Clamp to avoid log(0) when the booster is astronomically
            # confident against the winner on the holdout.
            total -= float(np.log(max(p_w, 1e-12)))
        return total

    # Golden-section search on log-T in [log 0.05, log 20] ≈ [-3, 3].
    a, b = -3.0, 3.0
    phi = (1 + 5**0.5) / 2
    resphi = 2 - phi
    tol = 1e-4
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = nll(float(np.exp(x1)))
    f2 = nll(float(np.exp(x2)))
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = nll(float(np.exp(x1)))
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = nll(float(np.exp(x2)))
    t_star = float(np.exp((a + b) / 2))
    return max(0.05, min(t_star, 20.0))


def _evaluate_ranker(
    model: lgb.Booster, frame: TrainingFrame,
) -> dict[str, float]:
    """Score a fitted booster against a labelled frame."""
    if frame.features.empty:
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "avg_winner_rank": 0.0,
            "n_races": 0,
        }

    scores = model.predict(frame.features)

    df = frame.features.copy()
    df["_score"] = scores
    df["_target"] = frame.target.values
    df["_race"] = frame.groups.values

    top1 = 0
    top3 = 0
    winner_ranks: list[int] = []
    for _race_id, race_df in df.groupby("_race", sort=False):
        ranked = race_df.sort_values("_score", ascending=False).reset_index(drop=True)
        # Winner = highest target in the race (ties broken by first appearance).
        winner_target = race_df["_target"].max()
        winners = ranked[ranked["_target"] == winner_target]
        if winners.empty:
            continue
        winner_rank = int(winners.index[0]) + 1  # 1-based
        winner_ranks.append(winner_rank)
        if winner_rank == 1:
            top1 += 1
        if winner_rank <= 3:
            top3 += 1

    n = len(winner_ranks)
    return {
        "top1_accuracy": (top1 / n) * 100 if n else 0.0,
        "top3_accuracy": (top3 / n) * 100 if n else 0.0,
        "avg_winner_rank": float(np.mean(winner_ranks)) if winner_ranks else 0.0,
        "n_races": n,
    }


def train_ranker(
    session: Session,
    *,
    from_date: date_type | None = None,
    to_date: date_type | None = None,
    holdout_fraction: float = 0.2,
    num_boost_round: int = 500,
    early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
    model_dir: Path | None = None,
    model_name: str | None = None,
    params: dict | None = None,
    exclude_features: list[str] | None = None,
) -> TrainingResult:
    """Fit a LightGBM LambdaRank model on resulted races.

    Parameters
    ----------
    from_date, to_date:
        Optional inclusive bounds on race dates.
    holdout_fraction:
        Fraction of the chronologically latest races reserved for the
        walk-forward test set.  The final model is fit on the train set
        with early stopping on the test set.
    num_boost_round:
        Maximum training rounds.  Early stopping may terminate sooner.
    params:
        LightGBM parameter overrides merged over the defaults.

    The trained booster is saved to ``<model_dir>/<name>.txt`` with a
    JSON sidecar (``<name>.meta.json``) capturing the feature column
    list, training bounds, and evaluation metrics.
    """
    effective_params = {**_DEFAULT_LGBM_PARAMS, **(params or {})}
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or DEFAULT_MODEL_BASENAME

    frame = build_training_frame(
        session, from_date=from_date, to_date=to_date,
    )
    if frame.features.empty:
        raise RuntimeError(
            "No AGF-bearing resulted races found for training. "
            "Run `ganyan scrape --results-range --from <date>` first.",
        )

    # Optionally strip specified columns before training.  Used by the
    # value-betting variant that must NOT see AGF during training so the
    # engineered features are forced to carry real weight.  The excluded
    # list is persisted so inference rebuilds the matrix the same way.
    excluded = list(exclude_features or [])
    if excluded:
        keep_cols = [c for c in FEATURE_COLUMNS if c not in excluded]
        frame = TrainingFrame(
            features=frame.features[keep_cols].reset_index(drop=True),
            target=frame.target,
            groups=frame.groups,
            race_dates=frame.race_dates,
        )
        logger.info(
            "Excluded %d feature(s) from training: %s",
            len(excluded), excluded,
        )

    train, test = _temporal_split(frame, holdout_fraction)

    train_dataset = lgb.Dataset(
        train.features,
        label=train.target,
        group=train.group_sizes(),
        free_raw_data=False,
    )
    valid_dataset = None
    callbacks = []
    if not test.features.empty:
        valid_dataset = lgb.Dataset(
            test.features,
            label=test.target,
            group=test.group_sizes(),
            reference=train_dataset,
            free_raw_data=False,
        )
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        )
    callbacks.append(lgb.log_evaluation(period=0))  # silence per-iter chatter

    booster = lgb.train(
        params=effective_params,
        train_set=train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[train_dataset] + ([valid_dataset] if valid_dataset else []),
        valid_names=["train"] + (["test"] if valid_dataset else []),
        callbacks=callbacks,
    )

    metrics = _evaluate_ranker(booster, test)
    # Fit the within-race softmax temperature on holdout.  Persisted
    # alongside the booster; MLPredictor reads it back and applies
    # ``raw / T`` before the softmax so downstream Kelly sizing gets
    # calibrated (not arbitrary-scaled) probabilities.
    temperature = _fit_temperature(booster, test)
    metrics["softmax_temperature"] = temperature

    # Save model + metadata.
    model_path = model_dir / f"{model_name}.txt"
    meta_path = model_dir / f"{model_name}.meta.json"
    booster.save_model(str(model_path))

    trained_feature_cols = [
        c for c in FEATURE_COLUMNS if c not in excluded
    ]
    importance_gain = booster.feature_importance(importance_type="gain")
    feature_importance = dict(
        sorted(
            zip(trained_feature_cols, importance_gain.tolist()),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    metadata = {
        "feature_columns": trained_feature_cols,
        "excluded_features": excluded,
        "params": effective_params,
        "train_races": int(train.groups.nunique()),
        "test_races": int(test.groups.nunique()),
        "train_rows": int(len(train.features)),
        "test_rows": int(len(test.features)),
        "holdout_fraction": holdout_fraction,
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "num_boost_round": num_boost_round,
        "best_iteration": booster.best_iteration or num_boost_round,
        "softmax_temperature": temperature,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info(
        "Trained LightGBM ranker — train_races=%d test_races=%d top1=%.2f%% top3=%.2f%%",
        metadata["train_races"],
        metadata["test_races"],
        metrics["top1_accuracy"],
        metrics["top3_accuracy"],
    )

    return TrainingResult(
        model_path=model_path,
        metadata_path=meta_path,
        train_races=metadata["train_races"],
        test_races=metadata["test_races"],
        metrics=metrics,
        feature_importance=feature_importance,
    )

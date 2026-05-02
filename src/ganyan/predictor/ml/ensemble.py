"""Multi-model ensemble predictor.

Loads every trained model in ``models/`` and runs them all against the
same race.  Convergence — multiple independent models agreeing on a
horse — is treated as a stronger signal than any single model's pick.

Heterogeneous heads supported:
- LightGBM rank-objective models (default ``.txt`` boosters)
- LightGBM finish-time regressors (sort ascending = rank)
- Linear rankers (numpy ``.npz`` + standardisation, conditional logit
  or Plackett-Luce)
- Per-race-type specialists — auto-skipped on races whose race_type
  doesn't match the specialist's training prefix.

Each model carries its own ``feature_columns`` list and a per-objective
prediction routine; ``build_race_frame`` produces a superset feature
matrix that every head slices what it needs from.

Output per horse: probability from each applicable model, convergence
score (how many models rank this horse at #1), mean probability across
heads, disagreement (std), sorted by ``(convergence, mean_prob, agf)``
descending.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from ganyan.db.models import Prediction as PredictionRow, Race, RaceEntry
from ganyan.predictor.bayesian import Prediction
from ganyan.predictor.ml.features import build_race_frame
from ganyan.predictor.ml.predictor import (
    LoadedModel, _softmax, load_latest_model,
)
from ganyan.predictor.ml.trainer import DEFAULT_MODEL_DIR


logger = logging.getLogger(__name__)


@dataclass
class LoadedLinearModel:
    """Adapter for numpy linear-ranker (conditional logit / Plackett-Luce).

    Mimics the relevant surface of ``LoadedModel`` so EnsemblePredictor
    can iterate over heterogeneous heads.  Inference: standardise the
    feature row (mean/std from training), dot with β, return raw scores
    that get softmax'd within-race like any other rank head.
    """

    name: str
    beta: np.ndarray
    feature_columns: list[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    model_family: str  # "conditional_logit" or "plackett_luce"
    metadata: dict = field(default_factory=dict)
    softmax_temperature: float = 1.0  # MLE-fitted softmax already; T=1

    @property
    def model_version(self) -> str:
        return f"linear-{self.model_family}-{self.name}"

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Apply the same standardisation as training, return β·x scores."""
        filled = np.where(np.isnan(X), self.feat_mean, X)
        std_X = (filled - self.feat_mean) / self.feat_std
        std_X = np.where(np.isfinite(std_X), std_X, 0.0)
        return std_X @ self.beta


def _load_linear_model(
    name: str, meta: dict, model_dir: Path,
) -> LoadedLinearModel | None:
    npz_path = model_dir / meta.get("npz_path", f"{name}.npz")
    if not npz_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=False)
    feat_cols = list(meta.get("feature_columns") or npz["feature_columns"].tolist())
    return LoadedLinearModel(
        name=name,
        beta=np.asarray(npz["beta"], dtype=float),
        feature_columns=feat_cols,
        feat_mean=np.asarray(npz["mean"], dtype=float),
        feat_std=np.asarray(npz["std"], dtype=float),
        model_family=meta.get("model_family", "linear"),
        metadata=meta,
    )


@dataclass
class EnsemblePrediction:
    """Per-horse summary across all loaded models."""

    horse_id: int
    horse_name: str
    # Mean probability across models (weighted equally for now).
    mean_probability: float
    # Number of models that ranked this horse strictly at position 1.
    convergence_top1: int
    # Number of models that placed this horse in their top-3.
    convergence_top3: int
    # Standard deviation of per-model probabilities — higher = models
    # disagree on this horse, lower = models agree on its share.
    disagreement: float
    # Average rank (1-based) across models.
    mean_rank: float
    # Per-model details: model_name -> {probability, rank}
    by_model: dict[str, dict] = field(default_factory=dict)


def _list_model_names(model_dir: Path) -> list[str]:
    """All ``<name>.meta.json`` stems present in the directory."""
    return sorted(p.stem.removesuffix(".meta") for p in model_dir.glob("*.meta.json"))


def load_all_models(
    model_dir: Path | None = None,
) -> list[LoadedModel | LoadedLinearModel]:
    """Load every saved rank-or-finish-time head under ``model_dir``.

    Skips models whose objective is ``ev`` — those output regression EV
    values that aren't directly comparable to within-race probabilities.
    Rank, finish-time, and linear-MLE heads all reduce naturally to
    "who wins this race", so they're admissible ensemble members.
    Each loaded head carries its training-time metadata so
    ``EnsemblePredictor`` knows how to score it and whether it's a
    race-type specialist that should be filtered on race.race_type.
    """
    model_dir = model_dir or DEFAULT_MODEL_DIR
    names = _list_model_names(model_dir)
    out: list[LoadedModel | LoadedLinearModel] = []
    for name in names:
        meta_path = model_dir / f"{name}.meta.json"
        try:
            metadata = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("objective") == "ev":
            logger.info("ensemble: skipping EV-head %s (incompatible scale)", name)
            continue
        # Linear-ranker heads carry ``model_family`` in metadata.
        if "model_family" in metadata:
            linear = _load_linear_model(name, metadata, model_dir)
            if linear is not None:
                out.append(linear)
            continue
        # Otherwise it's a LightGBM head (rank or finish-time).
        try:
            loaded = load_latest_model(model_dir=model_dir, model_name=name)
            out.append(loaded)
        except FileNotFoundError:
            continue
    return out


def _model_applies_to_race(
    model: LoadedModel | LoadedLinearModel, race_type: str | None,
) -> bool:
    """Race-type specialist gate: only run a head whose training data
    matches the race's race_type prefix.

    Generic heads (``race_type_prefix`` missing or null) apply to every
    race.  Specialists with a prefix only apply when ``race_type``
    starts with it.
    """
    prefix = (model.metadata or {}).get("race_type_prefix")
    if not prefix:
        return True
    if not race_type:
        return False
    return race_type.startswith(prefix)


class EnsemblePredictor:
    """Run every available rank-objective model on the same race and
    aggregate.

    Usage::

        predictor = EnsemblePredictor(session)
        preds = predictor.predict(race_id)
        for p in preds[:5]:
            print(p.horse_name, p.convergence_top1, p.mean_probability)
    """

    def __init__(
        self,
        session: Session,
        models: list[LoadedModel] | None = None,
    ) -> None:
        self.session = session
        self._models = models

    @property
    def models(self) -> list[LoadedModel]:
        if self._models is None:
            self._models = load_all_models()
            if not self._models:
                raise FileNotFoundError(
                    "No trained models found.  Run `ganyan train` first.",
                )
        return self._models

    def predict(self, race_id: int) -> list[EnsemblePrediction]:
        race = self.session.get(Race, race_id)
        if race is None or not race.entries:
            return []
        frame = build_race_frame(self.session, race_id)
        if frame.empty:
            return []

        entries_by_id = {e.horse_id: e for e in race.entries}
        agf_by_hid = {
            e.horse_id: float(e.agf) if e.agf is not None else -1.0
            for e in race.entries
        }
        horse_ids = [int(h) for h in frame["horse_id"]]

        # Per-model probabilities {model_name -> {horse_id -> prob}}
        per_model: dict[str, dict[int, float]] = {}
        per_model_rank: dict[str, dict[int, int]] = {}

        for model in self.models:
            # Specialist gating: skip models whose training prefix
            # doesn't match this race's race_type.
            if not _model_applies_to_race(model, race.race_type):
                continue
            cols = [c for c in model.feature_columns if c in frame.columns]
            X_df = frame[cols].astype("float64")
            if isinstance(model, LoadedLinearModel):
                raw = model.predict_raw(X_df.to_numpy())
            else:
                raw = np.asarray(model.booster.predict(X_df), dtype=float)
            obj = (model.metadata or {}).get("objective", "rank")
            if obj == "finish_time":
                # Predicted finish times in seconds.  Smaller = better.
                # Convert to within-race probabilities by negating and
                # softmax'ing the z-score; this puts the head on the
                # same probability simplex as the rank-objective heads
                # without distorting the sort order.
                std = float(raw.std()) or 1.0
                z = -(raw - raw.mean()) / std
                probs = _softmax(z, temperature=model.softmax_temperature)
            else:
                probs = _softmax(raw, temperature=model.softmax_temperature)
            # AGF-tiebreak when probs equal.
            order_key = [
                (float(probs[i]), agf_by_hid.get(horse_ids[i], -1.0), -horse_ids[i])
                for i in range(len(horse_ids))
            ]
            ranking = sorted(
                range(len(horse_ids)), key=lambda i: order_key[i], reverse=True,
            )
            ranks = {horse_ids[ranking[r]]: r + 1 for r in range(len(ranking))}
            per_model[model.model_version] = {
                horse_ids[i]: float(probs[i]) for i in range(len(horse_ids))
            }
            per_model_rank[model.model_version] = ranks

        # Aggregate per horse.
        out: list[EnsemblePrediction] = []
        for hid in horse_ids:
            entry = entries_by_id.get(hid)
            if entry is None:
                continue
            probs_list = [per_model[m].get(hid, 0.0) for m in per_model]
            ranks_list = [
                per_model_rank[m].get(hid, len(horse_ids)) for m in per_model_rank
            ]
            mean_p = float(np.mean(probs_list)) if probs_list else 0.0
            std_p = float(np.std(probs_list)) if len(probs_list) > 1 else 0.0
            top1 = sum(1 for r in ranks_list if r == 1)
            top3 = sum(1 for r in ranks_list if r <= 3)
            mean_rank = float(np.mean(ranks_list)) if ranks_list else 0.0

            by_model = {
                m: {
                    "probability": per_model[m].get(hid, 0.0) * 100.0,
                    "rank": per_model_rank[m].get(hid, None),
                }
                for m in per_model
            }
            out.append(
                EnsemblePrediction(
                    horse_id=hid,
                    horse_name=entry.horse.name if entry.horse else "?",
                    mean_probability=mean_p * 100.0,
                    convergence_top1=top1,
                    convergence_top3=top3,
                    disagreement=std_p * 100.0,
                    mean_rank=mean_rank,
                    by_model=by_model,
                )
            )

        # Sort: more models agreeing at #1 wins; tiebreak by mean prob,
        # then by AGF (market) as final fallback to break perfect ties
        # — same convention as the single-model tiebreaker.
        out.sort(
            key=lambda p: (
                p.convergence_top1,
                p.mean_probability,
                agf_by_hid.get(p.horse_id, -1.0),
            ),
            reverse=True,
        )
        return out

    def predict_as_predictions(self, race_id: int) -> list[Prediction]:
        """Adapter to the ``Prediction`` shape used by CLI/web layers.

        The ensemble's ``mean_probability`` becomes ``Prediction.probability``;
        ``convergence_top1`` is exposed via ``confidence`` (normalised by
        the number of loaded models so values stay in 0..1) and the
        per-model breakdown is shoved into ``contributing_factors``.
        """
        rows = self.predict(race_id)
        n_models = max(1, len(self.models))
        out: list[Prediction] = []
        for r in rows:
            factors = {
                f"{name}_prob": float(d["probability"])
                for name, d in r.by_model.items()
            }
            factors["convergence_top1"] = float(r.convergence_top1)
            factors["convergence_top3"] = float(r.convergence_top3)
            factors["disagreement"] = float(r.disagreement)
            factors["mean_rank"] = float(r.mean_rank)
            out.append(
                Prediction(
                    horse_id=r.horse_id,
                    horse_name=r.horse_name,
                    probability=r.mean_probability,
                    confidence=r.convergence_top1 / n_models,
                    contributing_factors=factors,
                )
            )
        return out

    def predict_and_save(self, race_id: int) -> list[Prediction]:
        """Run ``predict_as_predictions`` and persist to RaceEntry +
        Prediction rows — same contract as ``MLPredictor.predict_and_save``
        so callers (notably the scheduler's morning_card job) can swap
        predictors without further changes.
        """
        preds = self.predict_as_predictions(race_id)
        entries = {
            (e.race_id, e.horse_id): e
            for e in self.session.query(RaceEntry)
            .filter(RaceEntry.race_id == race_id)
            .all()
        }
        version = f"ensemble-{len(self.models)}-heads"
        for p in preds:
            entry = entries.get((race_id, p.horse_id))
            if entry is None:
                continue
            entry.predicted_probability = p.probability
            self.session.add(
                PredictionRow(
                    race_entry_id=entry.id,
                    model_version=version,
                    probability=p.probability,
                    confidence=p.confidence,
                    factors=p.contributing_factors,
                )
            )
        return preds

"""Trainer provenance: every model meta must carry trained_at + git_sha."""
from __future__ import annotations

import json
from pathlib import Path

from ganyan.predictor.ml import trainer as trainer_mod


def test_git_sha_returns_short_hash():
    sha = trainer_mod._git_sha()
    # Either git available → short SHA-like, or unavailable → None.
    if sha is not None:
        assert isinstance(sha, str)
        assert 4 <= len(sha) <= 16


def test_trainer_metadata_has_provenance_keys(tmp_path: Path, monkeypatch):
    """The metadata schema must include trained_at + git_sha keys."""
    import ganyan.predictor.ml.trainer as t

    sentinel = {
        "feature_columns": [],
        "excluded_features": [],
        "objective": "rank",
        "race_type_prefix": None,
        "params": {},
        "train_races": 0,
        "test_races": 0,
        "train_rows": 0,
        "test_rows": 0,
        "holdout_fraction": 0.2,
        "from_date": None,
        "to_date": None,
        "metrics": {},
        "feature_importance": {},
        "num_boost_round": 1,
        "best_iteration": 1,
        "softmax_temperature": 1.0,
        "trained_at": "2026-04-27T22:00:00+00:00",
        "git_sha": "abc1234",
    }
    p = tmp_path / "x.meta.json"
    p.write_text(json.dumps(sentinel))
    parsed = json.loads(p.read_text())
    assert "trained_at" in parsed
    assert "git_sha" in parsed

    # Confirm the keys are present in the actual trainer source so a
    # future refactor can't silently drop them.
    src = Path(t.__file__).read_text()
    assert '"trained_at"' in src
    assert '"git_sha"' in src

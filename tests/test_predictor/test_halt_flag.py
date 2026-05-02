import os
import time
import pytest
from pathlib import Path

from ganyan.predictor import halt_flag


@pytest.fixture
def tmp_flag(monkeypatch, tmp_path):
    p = tmp_path / "halt.flag"
    monkeypatch.setenv("GANYAN_HALT_FLAG_PATH", str(p))
    yield p


def test_no_flag_means_not_halted(tmp_flag):
    assert halt_flag.is_halted() is None


def test_set_halt_writes_file_and_returns_state(tmp_flag):
    halt_flag.set_halt(reason="rolling ROI -12% over 45 bets", source="rolling_pnl_halt")
    state = halt_flag.is_halted()
    assert state is not None
    assert state["reason"] == "rolling ROI -12% over 45 bets"
    assert state["source"] == "rolling_pnl_halt"
    assert "timestamp" in state


def test_set_halt_does_not_overwrite_existing_halt(tmp_flag):
    halt_flag.set_halt(reason="first reason", source="canary_a")
    halt_flag.set_halt(reason="second reason", source="canary_b")
    state = halt_flag.is_halted()
    assert state["reason"] == "first reason"
    assert state["source"] == "canary_a"


def test_clear_halt_removes_file(tmp_flag):
    halt_flag.set_halt(reason="x", source="y")
    assert halt_flag.is_halted() is not None
    halt_flag.clear_halt()
    assert halt_flag.is_halted() is None


def test_clear_halt_is_idempotent(tmp_flag):
    halt_flag.clear_halt()
    halt_flag.clear_halt()
    assert halt_flag.is_halted() is None


def test_corrupt_flag_returns_minimal_state(tmp_flag):
    tmp_flag.write_text("not json")
    state = halt_flag.is_halted()
    assert state is not None
    assert state["source"] == "unknown"

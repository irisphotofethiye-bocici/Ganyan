"""add multi_race_pools table

Captures higher-order TJK pari-mutuel pools that span multiple races
(5'lı / 6'lı / 7'lı GANYAN). The existing per-race payout columns on
``races`` only cover up to 4'lü. Multi-race pools are program-level,
not race-level, so they live in their own table keyed by
(date, track, pool_type, pool_index) — pool_index distinguishes the
"1. 6'LI" (races 1-6) from "2. 6'LI" (races 4-9) shape TJK runs on
some programs.

Revision ID: b9c0d1e2f3a4
Revises: a8b9c0d1e2f3
Create Date: 2026-05-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b9c0d1e2f3a4"
down_revision: Union[str, Sequence[str], None] = "a8b9c0d1e2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "multi_race_pools",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column(
            "track_id",
            sa.Integer,
            sa.ForeignKey("tracks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pool_type", sa.String(10), nullable=False),
        sa.Column("pool_index", sa.Integer, nullable=False, server_default="1"),
        sa.Column("start_race_no", sa.Integer, nullable=True),
        sa.Column("end_race_no", sa.Integer, nullable=True),
        sa.Column("winning_combo", sa.String(200), nullable=True),
        sa.Column("payout_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column(
            "captured_at",
            sa.DateTime,
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "date", "track_id", "pool_type", "pool_index",
            name="uq_multi_race_pool_date_track_type_idx",
        ),
    )
    op.create_index(
        "ix_multi_race_pools_date_track",
        "multi_race_pools", ["date", "track_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_multi_race_pools_date_track", "multi_race_pools")
    op.drop_table("multi_race_pools")

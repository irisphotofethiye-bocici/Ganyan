"""add multi_race_picks table

Stores generated 6'lı / 5'lı / 7'lı GANYAN coupons (which horses kept
per leg) plus their grading outcome once the multi_race_pool is
resulted. Mirrors the single-race ``picks`` table but keyed at the
program level since each row spans multiple races.

Revision ID: c0d1e2f3a4b5
Revises: b9c0d1e2f3a4
Create Date: 2026-05-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c0d1e2f3a4b5"
down_revision: Union[str, Sequence[str], None] = "b9c0d1e2f3a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "multi_race_picks",
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
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("start_race_no", sa.Integer, nullable=False),
        sa.Column("end_race_no", sa.Integer, nullable=False),
        # JSON: list of lists of horse program-numbers per leg.
        # e.g. [[1, 10], [4, 5, 11], [1, 2, 3, 4, 5, 7], ...]
        sa.Column("kept_horses_per_leg", sa.JSON, nullable=False),
        sa.Column("total_tickets", sa.Integer, nullable=False),
        sa.Column("ticket_unit_tl", sa.Numeric(10, 2), nullable=False),
        sa.Column("stake_tl", sa.Numeric(10, 2), nullable=False),
        # Per-leg model conviction (top-1 prob) at generation time, for
        # post-hoc analysis of why each leg was sized as it was.
        sa.Column("conviction_per_leg", sa.JSON, nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime,
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("graded", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("hit", sa.Boolean, nullable=True),
        sa.Column("payout_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column("net_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column("graded_at", sa.DateTime, nullable=True),
        sa.UniqueConstraint(
            "date", "track_id", "pool_type", "pool_index", "strategy",
            name="uq_multi_race_pick_date_track_type_idx_strat",
        ),
    )
    op.create_index(
        "ix_multi_race_picks_date_track",
        "multi_race_picks", ["date", "track_id"],
    )
    op.create_index(
        "ix_multi_race_picks_graded",
        "multi_race_picks", ["graded"],
    )


def downgrade() -> None:
    op.drop_index("ix_multi_race_picks_graded", "multi_race_picks")
    op.drop_index("ix_multi_race_picks_date_track", "multi_race_picks")
    op.drop_table("multi_race_picks")

"""add regime_daily table

Revision ID: a8b9c0d1e2f3
Revises: f2a3b4c5d6e7
Create Date: 2026-05-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, Sequence[str], None] = "f2a3b4c5d6e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "regime_daily",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("snapshot_date", sa.Date, nullable=False),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("n_winning", sa.Integer, nullable=False),
        sa.Column("mean_payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("mean_pool_proxy_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column("implied_takeout", sa.Numeric(6, 4), nullable=True),
        sa.Column("realized_vs_expected", sa.Numeric(6, 4), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "snapshot_date", "strategy", name="uq_regime_daily_date_strategy",
        ),
    )
    op.create_index(
        "ix_regime_daily_snapshot_date", "regime_daily", ["snapshot_date"],
    )
    op.create_index(
        "ix_regime_daily_strategy", "regime_daily", ["strategy"],
    )


def downgrade() -> None:
    op.drop_index("ix_regime_daily_strategy", "regime_daily")
    op.drop_index("ix_regime_daily_snapshot_date", "regime_daily")
    op.drop_table("regime_daily")

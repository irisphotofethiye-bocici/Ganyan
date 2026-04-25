"""add agf_snapshots table for late-money tracking

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-04-25

Captures AGF readings as a time-series so the predictor can use
late-money drift (the difference between AGF at card-publish and AGF
near post-time) as a feature.  Late drift is one of the few signals a
public-data scraper can extract that AGF itself doesn't already encode
(by definition, the AGF *value* is a single point in time — drift
requires multiple samples).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d0e1f2a3b4c5"
down_revision: Union[str, Sequence[str], None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agf_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "race_entry_id", sa.Integer(),
            sa.ForeignKey("race_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "taken_at", sa.DateTime(), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("agf", sa.Numeric(5, 2), nullable=False),
    )
    op.create_index(
        "ix_agf_snapshots_race_entry_id",
        "agf_snapshots", ["race_entry_id"],
    )
    op.create_index(
        "ix_agf_snapshots_taken_at",
        "agf_snapshots", ["taken_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_agf_snapshots_taken_at", table_name="agf_snapshots")
    op.drop_index("ix_agf_snapshots_race_entry_id", table_name="agf_snapshots")
    op.drop_table("agf_snapshots")

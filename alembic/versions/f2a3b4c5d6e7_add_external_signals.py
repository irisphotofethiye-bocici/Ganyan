"""add external_signals table

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-04-26

Stores arbitrary per-source signals collected from third-party racing
sites (tipster picks, reported jockeys, fixed-odds bookmaker prices)
in a single table.  Polymorphic by ``source_name`` + ``signal_type``;
``payload`` JSONB holds source-specific structured data the model
doesn't necessarily consume.  Foreign keys to race / race_entry are
nullable because some signals (race-level commentary, weather notes)
don't bind to a specific horse.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f2a3b4c5d6e7"
down_revision: Union[str, Sequence[str], None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "external_signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_name", sa.String(length=50), nullable=False),
        sa.Column("signal_type", sa.String(length=50), nullable=False),
        sa.Column(
            "race_id", sa.Integer(),
            sa.ForeignKey("races.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "race_entry_id", sa.Integer(),
            sa.ForeignKey("race_entries.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("value", sa.Numeric(10, 3), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column(
            "captured_at", sa.DateTime(), nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_external_signals_source_type",
        "external_signals", ["source_name", "signal_type"],
    )
    op.create_index(
        "ix_external_signals_race_entry_id",
        "external_signals", ["race_entry_id"],
    )
    op.create_index(
        "ix_external_signals_race_id",
        "external_signals", ["race_id"],
    )
    op.create_index(
        "ix_external_signals_captured_at",
        "external_signals", ["captured_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_external_signals_captured_at", table_name="external_signals")
    op.drop_index("ix_external_signals_race_id", table_name="external_signals")
    op.drop_index("ix_external_signals_race_entry_id", table_name="external_signals")
    op.drop_index("ix_external_signals_source_type", table_name="external_signals")
    op.drop_table("external_signals")

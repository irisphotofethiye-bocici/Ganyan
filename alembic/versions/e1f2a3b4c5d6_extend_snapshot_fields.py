"""extend agf_snapshots with jockey, equipment, gate fields

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-04-26

The AGF snapshot system already re-fetches the program every 30 min
during race hours.  Extending each snapshot row to ALSO record the
jockey, equipment, and gate at that time turns the same scrape into a
program-diff log: when the late snapshot's jockey differs from the
early snapshot's jockey, a last-minute jockey substitution happened —
typically because the regular jockey was reported (medical) or
penalized.  Same signal logic for equipment changes (first-time
blinkers added late = strong sürpriz-at indicator).

All three columns are nullable to preserve historical rows where this
information wasn't captured.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, Sequence[str], None] = "d0e1f2a3b4c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agf_snapshots",
        sa.Column("jockey", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "agf_snapshots",
        sa.Column("equipment", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "agf_snapshots",
        sa.Column("gate_number", sa.SmallInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agf_snapshots", "gate_number")
    op.drop_column("agf_snapshots", "equipment")
    op.drop_column("agf_snapshots", "jockey")

"""add race_entry.plase_payout_tl column

Revision ID: e2f3a4b5c6d7
Revises: d4e5f6a7b8c9
Create Date: 2026-05-06

Plase pool publishes per-horse payouts (TJK shows ``PLASE <program_no>
<amount>`` for each placed horse). We persist it on race_entries so
picks.grade_race can look up the bet's settled plase payout by the
horse we picked.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, Sequence[str], None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "race_entries",
        sa.Column("plase_payout_tl", sa.Numeric(8, 2), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("race_entries", "plase_payout_tl")

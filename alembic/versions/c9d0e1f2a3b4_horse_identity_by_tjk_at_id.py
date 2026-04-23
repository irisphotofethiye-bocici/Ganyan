"""switch horse identity from name-unique to tjk_at_id-unique

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-04-24

The prior schema enforced ``UNIQUE(name)`` on ``horses``.  Two distinct
horses with the same registered name at different tracks/eras would
collide into a single row, silently mixing their histories (sire,
trainer, age, race entries) — and corrupting every ``compute_*_win_rate``
lookup keyed on ``horse_id``.  TJK's ``AtId`` is the stable identity.

This migration drops the name-unique constraint and adds a partial
unique index on ``tjk_at_id`` (only when non-null, since a small
number of legacy rows predate the tjk_at_id crawl).
"""
from typing import Sequence, Union

from alembic import op


revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, Sequence[str], None] = "b8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("horses_name_key", "horses", type_="unique")
    # Partial unique index on tjk_at_id — safe for the ~85 legacy rows
    # with NULL tjk_at_id.  Once the horse-detail crawler has backfilled
    # tjk_at_id on all rows this could be tightened to a non-partial
    # UNIQUE, but the partial form keeps the migration non-destructive.
    op.create_index(
        "uq_horses_tjk_at_id",
        "horses",
        ["tjk_at_id"],
        unique=True,
        postgresql_where="tjk_at_id IS NOT NULL",
    )
    op.create_index("ix_horses_name", "horses", ["name"])


def downgrade() -> None:
    op.drop_index("ix_horses_name", table_name="horses")
    op.drop_index("uq_horses_tjk_at_id", table_name="horses")
    op.create_unique_constraint("horses_name_key", "horses", ["name"])

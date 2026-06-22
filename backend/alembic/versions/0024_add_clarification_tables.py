"""Add clarification_sessions and clarification_questions tables.

Supports the interactive ambiguity resolution workflow — sessions track
clarification exchanges, questions store per-field queries with candidate options.

Revision ID: 0024_add_clarification_tables
Revises: 0023_add_intent_outbox
Create Date: 2026-06-25 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0024_add_clarification_tables"
down_revision = "0023_add_intent_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types used by the clarification tables
    session_status_enum = postgresql.ENUM(
        "active", "resolved", "expired", "max_rounds_exceeded",
        name="session_status",
        create_type=False,
    )
    session_status_enum.create(op.get_bind(), checkfirst=True)

    clarification_outcome_enum = postgresql.ENUM(
        "RESOLVED", "STILL_AMBIGUOUS", "INVALID_RESPONSE",
        "CONFLICT_INTRODUCED", "MAX_ROUNDS_EXCEEDED", "SESSION_EXPIRED",
        name="clarification_outcome",
        create_type=False,
    )
    clarification_outcome_enum.create(op.get_bind(), checkfirst=True)

    reason_code_enum = postgresql.ENUM(
        "MULTIPLE_COLUMN_MATCHES", "AMBIGUOUS_REFERENCE",
        "LOW_CONFIDENCE_SCORE", "MISSING_COLUMN", "CONFLICTING_EVIDENCE",
        name="reason_code",
        create_type=False,
    )
    reason_code_enum.create(op.get_bind(), checkfirst=True)

    # -- clarification_sessions table --
    op.create_table(
        "clarification_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "submission_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("submissions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("intent_version", sa.Integer(), nullable=False),
        sa.Column("round_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_rounds", sa.Integer(), nullable=False, server_default="2"),
        sa.Column(
            "status",
            session_status_enum,
            nullable=False,
            server_default="active",
        ),
        sa.Column("revision_token", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("outcome", clarification_outcome_enum, nullable=True),
    )

    # Indexes for efficient lookups and expiration queries
    op.create_index(
        "ix_clarification_sessions_submission_id",
        "clarification_sessions",
        ["submission_id"],
    )
    op.create_index(
        "ix_clarification_sessions_status",
        "clarification_sessions",
        ["status"],
    )
    op.create_index(
        "ix_clarification_sessions_expires_at",
        "clarification_sessions",
        ["expires_at"],
    )

    # -- clarification_questions table --
    op.create_table(
        "clarification_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("clarification_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("round_number", sa.Integer(), nullable=False),
        sa.Column("intent_path", sa.Text(), nullable=False),
        sa.Column("reason_code", reason_code_enum, nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("candidate_options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("free_text_enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("user_answer", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("clarification_questions")
    op.drop_index("ix_clarification_sessions_expires_at", table_name="clarification_sessions")
    op.drop_index("ix_clarification_sessions_status", table_name="clarification_sessions")
    op.drop_index("ix_clarification_sessions_submission_id", table_name="clarification_sessions")
    op.drop_table("clarification_sessions")

    # Drop enum types (only if no other table uses them)
    op.execute("DROP TYPE IF EXISTS reason_code")
    op.execute("DROP TYPE IF EXISTS clarification_outcome")
    op.execute("DROP TYPE IF EXISTS session_status")

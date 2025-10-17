"""Deterministic UUID schema migration.

This migration intentionally drops the legacy INTEGER-based schema and rebuilds
all tables using the UUID layout defined in ``services.database.models``. It is
**destructive**: existing data will be removed. Run only when you have a backup
or explicitly want to reset the environment (e.g., fresh Railway deployment).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = "44bf430e49f1"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _drop_existing_schema(connection) -> None:
    """Drop known legacy tables if they exist."""
    drop_order = [
        "model_training_runs",
        "auto_accept_decisions",
        "auto_accept_rules",
        "annotation_events",
        "triage_items",
        "gold_annotations",
        "annotations",  # legacy table name from first schema
        "candidates",
        "sentences",
        "documents",
        "users",
    ]
    for table in drop_order:
        connection.execute(sa.text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))


def _column_defaults(dialect_name: str):
    if dialect_name == "postgresql":
        uuid_type = UUID(as_uuid=True)
        uuid_default = sa.text("uuid_generate_v4()")
        now_default = sa.func.now()
    else:
        uuid_type = sa.String(36)
        uuid_default = None
        now_default = sa.func.current_timestamp()
    return uuid_type, uuid_default, now_default


def upgrade() -> None:
    print("🚀 Starting deterministic UUID schema migration (all existing data will be dropped)")
    connection = op.get_bind()
    dialect_name = connection.dialect.name

    if dialect_name == "postgresql":
        print("📦 Ensuring uuid-ossp extension is available…")
        connection.execute(sa.text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))

    print("🗑️ Dropping legacy tables if they exist…")
    _drop_existing_schema(connection)

    uuid_type, uuid_default, now_default = _column_defaults(dialect_name)

    print("🏗️ Creating users table…")
    op.create_table(
        "users",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("username", sa.String(50), unique=True, nullable=False, index=True),
        sa.Column("email", sa.String(255), unique=True, index=True),
        sa.Column("hashed_password", sa.String(255)),
        sa.Column("role", sa.String(20), nullable=False, default="annotator"),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now_default, onupdate=now_default),
        sa.Column("last_login", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_user_active_role", "users", ["is_active", "role"])

    print("🏗️ Creating documents table…")
    op.create_table(
        "documents",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("doc_id", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500)),
        sa.Column("pub_date", sa.DateTime(timezone=True)),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("document_metadata", sa.JSON(), default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now_default, onupdate=now_default),
    )
    op.create_index("idx_document_source", "documents", ["source"])
    op.create_index("idx_document_created", "documents", ["created_at"])

    print("🏗️ Creating sentences table…")
    op.create_table(
        "sentences",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("sent_id", sa.String(100), nullable=False, index=True),
        sa.Column("document_id", uuid_type, sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("paragraph_id", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
    )
    op.create_index("idx_sentence_doc_sent", "sentences", ["document_id", "sent_id"])

    print("🏗️ Creating candidates table…")
    op.create_table(
        "candidates",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("sentence_id", uuid_type, sa.ForeignKey("sentences.id"), nullable=False),
        sa.Column("candidate_type", sa.String(20), nullable=False),
        sa.Column("text", sa.String(500)),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("start_offset", sa.Integer()),
        sa.Column("end_offset", sa.Integer()),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("head_candidate_id", uuid_type, sa.ForeignKey("candidates.id")),
        sa.Column("tail_candidate_id", uuid_type, sa.ForeignKey("candidates.id")),
        sa.Column("evidence", sa.Text()),
        sa.Column("score", sa.Float()),
        sa.Column("keywords", sa.JSON()),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50)),
        sa.Column("generation_method", sa.String(50), nullable=False),
        sa.Column("rule_pattern", sa.String(200)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
    )
    op.create_index("idx_candidate_confidence", "candidates", ["confidence"])
    op.create_index("idx_candidate_type", "candidates", ["candidate_type"])

    print("🏗️ Creating gold_annotations table…")
    op.create_table(
        "gold_annotations",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("document_id", uuid_type, sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("sentence_id", uuid_type, sa.ForeignKey("sentences.id"), nullable=False),
        sa.Column("entities", sa.JSON(), default=list),
        sa.Column("relations", sa.JSON(), default=list),
        sa.Column("topics", sa.JSON(), default=list),
        sa.Column("triplets", sa.JSON(), default=list),
        sa.Column("annotator_email", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("confidence_level", sa.String(20)),
        sa.Column("notes", sa.Text()),
        sa.Column("decision_method", sa.String(50)),
        sa.Column("source_candidate_id", uuid_type, sa.ForeignKey("candidates.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now_default, onupdate=now_default),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_annotation_status", "gold_annotations", ["status"])
    op.create_index("idx_annotation_doc_sent", "gold_annotations", ["document_id", "sentence_id"])

    print("🏗️ Creating triage_items table…")
    op.create_table(
        "triage_items",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("item_id", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("sentence_id", uuid_type, sa.ForeignKey("sentences.id"), nullable=False),
        sa.Column("candidate_id", uuid_type, sa.ForeignKey("candidates.id"), nullable=False),
        sa.Column("priority_score", sa.Float(), nullable=False),
        sa.Column("priority_level", sa.String(20), nullable=False),
        sa.Column("confidence_score", sa.Float(), default=0.0),
        sa.Column("novelty_score", sa.Float(), default=0.0),
        sa.Column("impact_score", sa.Float(), default=0.0),
        sa.Column("disagreement_score", sa.Float(), default=0.0),
        sa.Column("authority_score", sa.Float(), default=0.0),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("assigned_to", sa.String(255)),
        sa.Column("assigned_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now_default, onupdate=now_default),
    )
    op.create_index("idx_triage_priority", "triage_items", ["priority_score"])
    op.create_index("idx_triage_status", "triage_items", ["status"])

    print("🏗️ Creating annotation_events table…")
    op.create_table(
        "annotation_events",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("event_id", sa.String(255), unique=True, nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("triage_item_id", uuid_type, sa.ForeignKey("triage_items.id")),
        sa.Column("annotator_email", sa.String(255), nullable=False),
        sa.Column("processing_time", sa.Float()),
        sa.Column("decision", sa.String(50)),
        sa.Column("document_metadata", sa.JSON(), default=dict),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=now_default, index=True),
    )

    print("🏗️ Creating auto_accept_rules table…")
    op.create_table(
        "auto_accept_rules",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("rule_id", sa.String(100), unique=True, nullable=False),
        sa.Column("rule_name", sa.String(200), nullable=False),
        sa.Column("entity_types", sa.JSON()),
        sa.Column("relation_types", sa.JSON()),
        sa.Column("min_confidence", sa.Float(), default=0.95),
        sa.Column("min_agreement", sa.Float(), default=0.9),
        sa.Column("source_authority_min", sa.Float(), default=0.8),
        sa.Column("requires_rule_support", sa.Boolean(), default=True),
        sa.Column("max_novelty", sa.Float(), default=0.3),
        sa.Column("enabled", sa.Boolean(), default=True),
        sa.Column("auto_accepted", sa.Integer(), default=0),
        sa.Column("false_positives", sa.Integer(), default=0),
        sa.Column("precision", sa.Float(), default=1.0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=now_default, onupdate=now_default),
    )

    print("🏗️ Creating auto_accept_decisions table…")
    op.create_table(
        "auto_accept_decisions",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("triage_item_id", uuid_type, sa.ForeignKey("triage_items.id"), nullable=False),
        sa.Column("rule_id", uuid_type, sa.ForeignKey("auto_accept_rules.id")),
        sa.Column("decision", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reasoning", sa.Text()),
        sa.Column("was_correct", sa.Boolean()),
        sa.Column("validated_by", sa.String(255)),
        sa.Column("validated_at", sa.DateTime(timezone=True)),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=now_default),
    )

    print("🏗️ Creating model_training_runs table…")
    op.create_table(
        "model_training_runs",
        sa.Column("id", uuid_type, primary_key=True, server_default=uuid_default),
        sa.Column("run_id", sa.String(100), unique=True, nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("training_data_version", sa.String(50)),
        sa.Column("gold_annotation_count", sa.Integer()),
        sa.Column("trigger_conditions", sa.JSON()),
        sa.Column("confidence_score", sa.Float()),
        sa.Column("data_snapshot", sa.JSON()),
        sa.Column("triggered_by", sa.String(255)),
        sa.Column("hyperparameters", sa.JSON(), default=dict),
        sa.Column("training_config", sa.JSON(), default=dict),
        sa.Column("train_metrics", sa.JSON(), default=dict),
        sa.Column("val_metrics", sa.JSON(), default=dict),
        sa.Column("test_metrics", sa.JSON(), default=dict),
        sa.Column("status", sa.String(50), nullable=False, default="initiated"),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("error_message", sa.Text()),
        sa.Column("model_path", sa.String(500)),
        sa.Column("model_version", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now_default),
    )

    print("✅ UUID schema created successfully – database is ready for the new deployment")


def downgrade() -> None:
    print("⚠️ Downgrading: dropping UUID schema (all data will be removed)")
    connection = op.get_bind()
    _drop_existing_schema(connection)
    print("⚠️ Legacy INTEGER schema is not recreated automatically. Restore from backup if needed.")

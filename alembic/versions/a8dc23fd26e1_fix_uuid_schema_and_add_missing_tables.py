"""fix_uuid_schema_and_add_missing_tables

Revision ID: a8dc23fd26e1
Revises: eb64a6ba0005
Create Date: 2025-10-16 20:36:11.742268

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a8dc23fd26e1'
down_revision: Union[str, Sequence[str], None] = 'eb64a6ba0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Fix UUID schema and add missing tables.
    
    This migration:
    1. Enables UUID extension
    2. Creates missing tables with UUID schema
    3. Migrates existing tables from INTEGER to UUID
    4. Updates all foreign key relationships
    """
    # Enable UUID extension for PostgreSQL
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # Import UUID for column definitions
    from sqlalchemy.dialects.postgresql import UUID
    
    # STEP 1: Add missing critical tables with correct UUID schema
    print("Creating missing tables with UUID schema...")
    
    # Create triage_items table (CRITICAL - this is why documents don't appear in queue!)
    op.create_table('triage_items',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('item_id', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('sentence_id', UUID(as_uuid=True), nullable=False),
        sa.Column('candidate_id', UUID(as_uuid=True), nullable=False),
        sa.Column('priority_score', sa.Float(), nullable=False),
        sa.Column('priority_level', sa.String(20), nullable=False),
        sa.Column('confidence_score', sa.Float(), default=0.0),
        sa.Column('novelty_score', sa.Float(), default=0.0),
        sa.Column('impact_score', sa.Float(), default=0.0),
        sa.Column('disagreement_score', sa.Float(), default=0.0),
        sa.Column('authority_score', sa.Float(), default=0.0),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('assigned_to', sa.String(255)),
        sa.Column('assigned_at', sa.DateTime()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Create annotation_events table (EXACT match to models.py:160-179)
    op.create_table('annotation_events',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('event_id', sa.String(255), unique=True, nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('triage_item_id', UUID(as_uuid=True)),
        sa.Column('annotator_email', sa.String(255), nullable=False),
        sa.Column('processing_time', sa.Float()),
        sa.Column('decision', sa.String(50)),
        sa.Column('document_metadata', sa.JSON(), default={}),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), index=True)
    )
    
    # Create auto_accept_rules table (EXACT match to models.py:181-207)
    op.create_table('auto_accept_rules',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('rule_id', sa.String(100), unique=True, nullable=False),
        sa.Column('rule_name', sa.String(200), nullable=False),
        sa.Column('entity_types', sa.JSON()),
        sa.Column('relation_types', sa.JSON()),
        sa.Column('min_confidence', sa.Float(), default=0.95),
        sa.Column('min_agreement', sa.Float(), default=0.9),
        sa.Column('source_authority_min', sa.Float(), default=0.8),
        sa.Column('requires_rule_support', sa.Boolean(), default=True),
        sa.Column('max_novelty', sa.Float(), default=0.3),
        sa.Column('enabled', sa.Boolean(), default=True),
        sa.Column('auto_accepted', sa.Integer(), default=0),
        sa.Column('false_positives', sa.Integer(), default=0),
        sa.Column('precision', sa.Float(), default=1.0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Create auto_accept_decisions table (EXACT match to models.py:209-226)
    op.create_table('auto_accept_decisions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('triage_item_id', UUID(as_uuid=True), nullable=False),
        sa.Column('rule_id', UUID(as_uuid=True)),
        sa.Column('decision', sa.String(50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('reasoning', sa.Text()),
        sa.Column('was_correct', sa.Boolean()),
        sa.Column('validated_by', sa.String(255)),
        sa.Column('validated_at', sa.DateTime()),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now())
    )
    
    # Create model_training_runs table (EXACT match to models.py:228-265)
    op.create_table('model_training_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('run_id', sa.String(100), unique=True, nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('training_data_version', sa.String(50)),
        sa.Column('gold_annotation_count', sa.Integer()),
        sa.Column('trigger_conditions', sa.JSON()),
        sa.Column('confidence_score', sa.Float()),
        sa.Column('data_snapshot', sa.JSON()),
        sa.Column('triggered_by', sa.String(255)),
        sa.Column('hyperparameters', sa.JSON(), default={}),
        sa.Column('training_config', sa.JSON(), default={}),
        sa.Column('train_metrics', sa.JSON(), default={}),
        sa.Column('val_metrics', sa.JSON(), default={}),
        sa.Column('test_metrics', sa.JSON(), default={}),
        sa.Column('status', sa.String(50), nullable=False, default='initiated'),
        sa.Column('started_at', sa.DateTime()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('error_message', sa.Text()),
        sa.Column('model_path', sa.String(500)),
        sa.Column('model_version', sa.String(50)),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    
    print("Missing tables created successfully!")
    
    # STEP 2: Migrate existing tables from INTEGER to UUID
    print("Starting INTEGER to UUID migration for existing tables...")
    
    # Migration order: independent tables first, then dependent tables
    tables_to_migrate = [
        # Independent tables first
        ('users', []),
        ('documents', []),
        # Dependent tables
        ('sentences', [('document_id', 'documents', 'id')]),
        ('candidates', [
            ('sentence_id', 'sentences', 'id'),
            ('head_candidate_id', 'candidates', 'id'),  # self-reference
            ('tail_candidate_id', 'candidates', 'id')   # self-reference
        ]),
        ('annotations', [
            ('document_id', 'documents', 'id'),
            ('sentence_id', 'sentences', 'id'),
            ('candidate_id', 'candidates', 'id'),
            ('user_id', 'users', 'id'),
            ('superseded_by', 'annotations', 'id')  # self-reference
        ])
    ]
    
    # Store mapping from old INTEGER IDs to new UUIDs for each table
    id_mappings = {}
    
    for table_name, foreign_keys in tables_to_migrate:
        print(f"Migrating table: {table_name}")
        
        # Add temporary UUID column
        op.add_column(table_name, sa.Column('new_id', UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()')))
        
        # Get existing data with mappings
        connection = op.get_bind()
        result = connection.execute(sa.text(f"SELECT id, new_id FROM {table_name}"))
        id_mappings[table_name] = {row[0]: row[1] for row in result.fetchall()}
        print(f"  - Mapped {len(id_mappings[table_name])} existing records")
        
        # Add new UUID foreign key columns
        for fk_column, ref_table, ref_column in foreign_keys:
            if fk_column in ['head_candidate_id', 'tail_candidate_id', 'superseded_by']:
                # Self-referential or nullable FKs
                op.add_column(table_name, sa.Column(f'new_{fk_column}', UUID(as_uuid=True)))
            else:
                op.add_column(table_name, sa.Column(f'new_{fk_column}', UUID(as_uuid=True)))
        
        # Update foreign key values using mappings
        for fk_column, ref_table, ref_column in foreign_keys:
            print(f"  - Updating foreign key: {fk_column} -> {ref_table}.{ref_column}")
            if ref_table in id_mappings:
                for old_id, new_id in id_mappings[ref_table].items():
                    connection.execute(sa.text(f"""
                        UPDATE {table_name} 
                        SET new_{fk_column} = :new_id 
                        WHERE {fk_column} = :old_id
                    """), {"new_id": new_id, "old_id": old_id})
    
    print("Data migration completed!")
    
    # STEP 3: Drop old constraints and columns, rename new ones
    print("Updating schema structure...")
    
    for table_name, foreign_keys in reversed(tables_to_migrate):
        print(f"Finalizing table: {table_name}")
        
        # Drop foreign key constraints first
        for fk_column, ref_table, ref_column in foreign_keys:
            try:
                # Find and drop FK constraint (name might vary)
                constraint_query = sa.text(f"""
                    SELECT constraint_name FROM information_schema.table_constraints 
                    WHERE table_name = '{table_name}' 
                    AND constraint_type = 'FOREIGN KEY'
                    AND constraint_name LIKE '%{fk_column}%'
                """)
                constraint_result = connection.execute(constraint_query).fetchall()
                for (constraint_name,) in constraint_result:
                    op.drop_constraint(constraint_name, table_name, type_='foreignkey')
            except Exception as e:
                print(f"    Warning: Could not drop FK constraint for {fk_column}: {e}")
        
        # Drop old columns
        for fk_column, _, _ in foreign_keys:
            op.drop_column(table_name, fk_column)
        
        # Drop old primary key constraint and column
        try:
            op.drop_constraint(f'{table_name}_pkey', table_name, type_='primary')
        except Exception as e:
            print(f"    Warning: Could not drop PK constraint: {e}")
        
        op.drop_column(table_name, 'id')
        
        # Rename new columns
        op.alter_column(table_name, 'new_id', new_column_name='id')
        for fk_column, _, _ in foreign_keys:
            op.alter_column(table_name, f'new_{fk_column}', new_column_name=fk_column)
        
        # Add new primary key constraint
        op.create_primary_key(f'{table_name}_pkey', table_name, ['id'])
    
    # STEP 4: Add foreign key constraints for migrated tables
    print("Adding foreign key constraints...")
    
    for table_name, foreign_keys in tables_to_migrate:
        for fk_column, ref_table, ref_column in foreign_keys:
            if fk_column not in ['head_candidate_id', 'tail_candidate_id', 'superseded_by']:
                # Required foreign keys
                op.create_foreign_key(
                    f'fk_{table_name}_{fk_column}', 
                    table_name, ref_table, 
                    [fk_column], [ref_column]
                )
            else:
                # Optional/self-referential foreign keys
                op.create_foreign_key(
                    f'fk_{table_name}_{fk_column}', 
                    table_name, ref_table, 
                    [fk_column], [ref_column]
                )
    
    # STEP 5: Add foreign key constraints for new tables
    print("Adding foreign key constraints for new tables...")
    
    # triage_items foreign keys (will reference migrated tables)
    op.create_foreign_key('fk_triage_items_sentence_id', 'triage_items', 'sentences', ['sentence_id'], ['id'])
    op.create_foreign_key('fk_triage_items_candidate_id', 'triage_items', 'candidates', ['candidate_id'], ['id'])
    
    # annotation_events foreign keys
    op.create_foreign_key('fk_annotation_events_triage_item_id', 'annotation_events', 'triage_items', ['triage_item_id'], ['id'])
    
    # auto_accept_decisions foreign keys
    op.create_foreign_key('fk_auto_accept_decisions_triage_item_id', 'auto_accept_decisions', 'triage_items', ['triage_item_id'], ['id'])
    op.create_foreign_key('fk_auto_accept_decisions_rule_id', 'auto_accept_decisions', 'auto_accept_rules', ['rule_id'], ['id'])
    
    # STEP 6: Fix table name mismatch (annotations -> gold_annotations)
    print("Fixing table name mismatch...")
    op.rename_table('annotations', 'gold_annotations')
    
    # Update constraint names for renamed table
    op.execute("ALTER INDEX annotations_pkey RENAME TO gold_annotations_pkey")
    
    # STEP 7: Add missing columns to existing tables that were missed
    print("Adding missing columns...")
    
    # Add missing columns to candidates table to match model
    existing_columns = [
        'text', 'label', 'start_offset', 'end_offset', 
        'evidence', 'score', 'keywords', 'model_name', 
        'model_version', 'generation_method', 'rule_pattern'
    ]
    
    for column in existing_columns:
        try:
            if column in ['text', 'label']:
                op.add_column('candidates', sa.Column(column, sa.String(500)))
            elif column in ['start_offset', 'end_offset']:
                op.add_column('candidates', sa.Column(column, sa.Integer()))
            elif column == 'evidence':
                op.add_column('candidates', sa.Column(column, sa.Text()))
            elif column == 'score':
                op.add_column('candidates', sa.Column(column, sa.Float()))
            elif column == 'keywords':
                op.add_column('candidates', sa.Column(column, sa.JSON()))
            elif column in ['model_name', 'generation_method']:
                op.add_column('candidates', sa.Column(column, sa.String(100)))
            elif column in ['model_version', 'rule_pattern']:
                op.add_column('candidates', sa.Column(column, sa.String(200)))
        except Exception as e:
            print(f"    Column {column} may already exist or have different constraints: {e}")
    
    # Convert existing paragraph_id from VARCHAR to INTEGER (as per models.py:47)
    try:
        # Check if paragraph_id exists and convert type
        connection = op.get_bind()
        result = connection.execute(sa.text("""
            SELECT data_type FROM information_schema.columns 
            WHERE table_name = 'sentences' AND column_name = 'paragraph_id'
        """)).fetchone()
        
        if result and result[0] != 'integer':
            print("    Converting paragraph_id from VARCHAR to INTEGER...")
            op.execute("ALTER TABLE sentences ALTER COLUMN paragraph_id TYPE INTEGER USING paragraph_id::INTEGER")
        elif not result:
            print("    Adding paragraph_id INTEGER column...")
            op.add_column('sentences', sa.Column('paragraph_id', sa.Integer()))
        else:
            print("    paragraph_id already correct type")
    except Exception as e:
        print(f"    paragraph_id column handling warning: {e}")
    
    # Rename document metadata column to match model
    try:
        op.alter_column('documents', 'doc_metadata', new_column_name='document_metadata')
    except Exception as e:
        print(f"    Column rename may have failed: {e}")
    
    print("Migration completed successfully!")
    print("🎉 The triage_items table now exists - documents should appear in queue!")


def downgrade() -> None:
    """
    Downgrade schema - DANGEROUS: This will convert UUIDs back to INTEGERs and may cause data loss!
    
    This rollback:
    1. Drops new tables
    2. Reverts UUID columns back to INTEGER
    3. Restores original schema
    
    WARNING: This may result in data loss if UUID values cannot be converted back!
    """
    print("WARNING: Starting dangerous downgrade - this may cause data loss!")
    
    # Import required types
    from sqlalchemy.dialects.postgresql import UUID
    
    # STEP 1: Drop all new tables (in reverse dependency order)
    print("Dropping new tables...")
    tables_to_drop = [
        'auto_accept_decisions',
        'annotation_events', 
        'model_training_runs',
        'auto_accept_rules',
        'triage_items'
    ]
    
    for table in tables_to_drop:
        try:
            op.drop_table(table)
            print(f"  - Dropped table: {table}")
        except Exception as e:
            print(f"  - Warning: Could not drop {table}: {e}")
    
    # STEP 2: Rename gold_annotations back to annotations
    print("Renaming gold_annotations back to annotations...")
    try:
        op.rename_table('gold_annotations', 'annotations')
        op.execute("ALTER INDEX gold_annotations_pkey RENAME TO annotations_pkey")
    except Exception as e:
        print(f"  - Warning: Table rename failed: {e}")
    
    # STEP 3: Convert UUID columns back to INTEGER (DANGEROUS!)
    print("Converting UUID back to INTEGER - THIS MAY CAUSE DATA LOSS!")
    
    connection = op.get_bind()
    
    # Tables to convert back (in reverse order)
    tables_to_revert = [
        ('annotations', [
            ('document_id', 'documents', 'id'),
            ('sentence_id', 'sentences', 'id'), 
            ('candidate_id', 'candidates', 'id'),
            ('user_id', 'users', 'id'),
            ('superseded_by', 'annotations', 'id')
        ]),
        ('candidates', [
            ('sentence_id', 'sentences', 'id'),
            ('head_candidate_id', 'candidates', 'id'),
            ('tail_candidate_id', 'candidates', 'id')
        ]),
        ('sentences', [('document_id', 'documents', 'id')]),
        ('documents', []),
        ('users', [])
    ]
    
    # Create mapping from UUID to sequential INTEGER for each table
    id_mappings = {}
    
    for table_name, foreign_keys in tables_to_revert:
        print(f"Converting {table_name} from UUID to INTEGER...")
        
        # Get all current UUID values
        result = connection.execute(sa.text(f"SELECT id FROM {table_name} ORDER BY created_at"))
        uuid_ids = [row[0] for row in result.fetchall()]
        
        # Create INTEGER mapping (1, 2, 3, ...)
        id_mappings[table_name] = {uuid_id: i+1 for i, uuid_id in enumerate(uuid_ids)}
        print(f"  - Will map {len(id_mappings[table_name])} UUIDs to INTEGERs")
        
        # Add temporary INTEGER columns
        op.add_column(table_name, sa.Column('temp_id', sa.Integer()))
        for fk_column, _, _ in foreign_keys:
            op.add_column(table_name, sa.Column(f'temp_{fk_column}', sa.Integer()))
        
        # Update with INTEGER values
        for uuid_id, int_id in id_mappings[table_name].items():
            connection.execute(sa.text(f"""
                UPDATE {table_name} SET temp_id = :int_id WHERE id = :uuid_id
            """), {"int_id": int_id, "uuid_id": uuid_id})
        
        # Update foreign keys
        for fk_column, ref_table, _ in foreign_keys:
            if ref_table in id_mappings:
                for uuid_id, int_id in id_mappings[ref_table].items():
                    connection.execute(sa.text(f"""
                        UPDATE {table_name} SET temp_{fk_column} = :int_id WHERE {fk_column} = :uuid_id
                    """), {"int_id": int_id, "uuid_id": uuid_id})
    
    # Drop constraints and rename columns back
    for table_name, foreign_keys in tables_to_revert:
        print(f"Finalizing {table_name} conversion...")
        
        # Drop foreign keys
        for fk_column, _, _ in foreign_keys:
            try:
                constraint_query = sa.text(f"""
                    SELECT constraint_name FROM information_schema.table_constraints 
                    WHERE table_name = '{table_name}' AND constraint_type = 'FOREIGN KEY'
                    AND constraint_name LIKE '%{fk_column}%'
                """)
                constraints = connection.execute(constraint_query).fetchall()
                for (constraint_name,) in constraints:
                    op.drop_constraint(constraint_name, table_name, type_='foreignkey')
            except Exception as e:
                print(f"    Warning: FK drop failed for {fk_column}: {e}")
        
        # Drop primary key and UUID columns
        try:
            op.drop_constraint(f'{table_name}_pkey', table_name, type_='primary')
        except Exception as e:
            print(f"    Warning: PK drop failed: {e}")
        
        op.drop_column(table_name, 'id')
        for fk_column, _, _ in foreign_keys:
            op.drop_column(table_name, fk_column)
        
        # Rename temp columns
        op.alter_column(table_name, 'temp_id', new_column_name='id')
        for fk_column, _, _ in foreign_keys:
            op.alter_column(table_name, f'temp_{fk_column}', new_column_name=fk_column)
        
        # Add primary key back
        op.create_primary_key(f'{table_name}_pkey', table_name, ['id'])
    
    # Restore foreign key constraints
    for table_name, foreign_keys in reversed(tables_to_revert):
        for fk_column, ref_table, ref_column in foreign_keys:
            try:
                op.create_foreign_key(
                    f'fk_{table_name}_{fk_column}',
                    table_name, ref_table,
                    [fk_column], [ref_column]
                )
            except Exception as e:
                print(f"    Warning: FK restore failed for {fk_column}: {e}")
    
    # Remove UUID extension
    try:
        op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
    except Exception as e:
        print(f"    Warning: Could not drop UUID extension: {e}")
    
    print("Downgrade completed - schema reverted to INTEGER IDs")
    print("⚠️  WARNING: Any data with UUID references may have been lost!")

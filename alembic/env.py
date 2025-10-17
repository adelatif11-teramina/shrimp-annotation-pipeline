import os
import sys
import logging
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, create_engine, text
from sqlalchemy import pool
from dotenv import load_dotenv

from alembic import context

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Configure logging for detailed migration tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ALEMBIC - %(levelname)s - %(message)s'
)
logger = logging.getLogger('alembic.env')

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# CRITICAL: Get database URL and validate it exists
database_url = os.getenv("DATABASE_URL")

if not database_url:
    error_msg = "❌ CRITICAL: DATABASE_URL environment variable not found! Cannot run migrations without database connection."
    logger.error(error_msg)
    raise ValueError(error_msg)

# Fix Railway's postgres:// URL format to postgresql:// for SQLAlchemy
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    logger.info("🔧 Converted postgres:// URL to postgresql:// for SQLAlchemy compatibility")

logger.info(f"🔗 Using database URL: {database_url[:50]}...")

# Test database connection before proceeding
try:
    test_engine = create_engine(database_url)
    with test_engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection test successful")
    test_engine.dispose()
except Exception as db_error:
    error_msg = f"❌ CRITICAL: Database connection failed: {db_error}"
    logger.error(error_msg)
    raise ConnectionError(error_msg) from db_error

config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
try:
    from services.database.models import Base
    target_metadata = Base.metadata
    logger.info(f"✅ Successfully imported models with {len(Base.metadata.tables)} tables")
    logger.info(f"📋 Tables: {sorted(Base.metadata.tables.keys())}")
    
    # Validate critical tables exist
    required_tables = ['documents', 'sentences', 'candidates', 'triage_items', 'gold_annotations']
    missing_tables = [t for t in required_tables if t not in Base.metadata.tables]
    if missing_tables:
        error_msg = f"❌ CRITICAL: Missing required tables in models: {missing_tables}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate UUID usage in critical tables
    for table_name in ['documents', 'sentences', 'candidates']:
        table = Base.metadata.tables[table_name]
        id_column = table.c.id
        if 'UUID' not in str(id_column.type):
            error_msg = f"❌ CRITICAL: Table {table_name}.id is not UUID type: {id_column.type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    logger.info("✅ Models validation successful - all tables use UUID schema")
    
except ImportError as import_error:
    error_msg = f"❌ CRITICAL: Failed to import models: {import_error}"
    logger.error(error_msg)
    raise ImportError(error_msg) from import_error

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    logger.info("🔄 Starting offline migrations...")
    url = config.get_main_option("sqlalchemy.url")
    
    try:
        context.configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            render_as_batch=True,  # Enable batch operations for SQLite compatibility
        )

        with context.begin_transaction():
            logger.info("🔄 Running migrations in offline mode...")
            context.run_migrations()
            logger.info("✅ Offline migrations completed successfully")
            
    except Exception as migration_error:
        error_msg = f"❌ CRITICAL: Offline migration failed: {migration_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from migration_error


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    logger.info("🔄 Starting online migrations...")
    
    try:
        # Configure engine with PostgreSQL-specific settings
        configuration = config.get_section(config.config_ini_section, {})
        configuration['sqlalchemy.poolclass'] = pool.NullPool
        
        # Add PostgreSQL-specific configurations for Railway
        if database_url.startswith("postgresql://"):
            logger.info("🐘 Configuring PostgreSQL-specific settings for Railway...")
            configuration.update({
                'sqlalchemy.pool_pre_ping': True,  # Validate connections
                'sqlalchemy.pool_recycle': 300,    # Recycle connections every 5 minutes  
                'sqlalchemy.echo': False,          # Disable SQL echo for production
            })
        
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            logger.info("🔗 Database connection established for migrations")
            
            context.configure(
                connection=connection, 
                target_metadata=target_metadata,
                render_as_batch=True,  # Enable batch operations
                compare_type=True,     # Compare column types
                compare_server_default=True,  # Compare default values
            )

            with context.begin_transaction():
                # Check current database state
                migration_context = context.get_context()
                current_revision = migration_context.get_current_revision()
                logger.info(f"📋 Current database revision: {current_revision}")
                
                logger.info("🔄 Running migrations in online mode...")
                context.run_migrations()
                
                # Verify migration success
                new_revision = migration_context.get_current_revision()
                logger.info(f"📋 New database revision: {new_revision}")
                logger.info("✅ Online migrations completed successfully")
                
    except Exception as migration_error:
        error_msg = f"❌ CRITICAL: Online migration failed: {migration_error}"
        logger.error(error_msg)
        logger.error(f"Database URL (truncated): {database_url[:50]}...")
        raise RuntimeError(error_msg) from migration_error


if context.is_offline_mode():
    logger.info("🔄 Alembic running in OFFLINE mode")
    run_migrations_offline()
else:
    logger.info("🔄 Alembic running in ONLINE mode")
    run_migrations_online()

logger.info("🎉 Alembic execution completed successfully!")

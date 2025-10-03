#!/usr/bin/env python3
"""
Database Management Script
Handles migrations, setup, and maintenance
"""

import os
import sys
import argparse
from pathlib import Path
from alembic.config import Config
from alembic import command
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

def get_alembic_config():
    """Get Alembic configuration"""
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))
    
    # Set database URL from environment
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/local/annotations.db")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    return alembic_cfg

def init_database():
    """Initialize database with latest schema"""
    print("🏗️  Initializing database...")
    
    # Ensure data directory exists
    data_dir = project_root / "data" / "local"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    alembic_cfg = get_alembic_config()
    
    try:
        # Upgrade to latest
        command.upgrade(alembic_cfg, "head")
        print("✅ Database initialized successfully")
        
        # Show current revision
        command.current(alembic_cfg)
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False
    
    return True

def create_migration(message: str):
    """Create a new migration"""
    print(f"📝 Creating migration: {message}")
    
    alembic_cfg = get_alembic_config()
    
    try:
        command.revision(alembic_cfg, message=message, autogenerate=True)
        print("✅ Migration created successfully")
    except Exception as e:
        print(f"❌ Failed to create migration: {e}")
        return False
    
    return True

def upgrade_database(target="head"):
    """Upgrade database to target revision"""
    print(f"⬆️  Upgrading database to {target}...")
    
    alembic_cfg = get_alembic_config()
    
    try:
        command.upgrade(alembic_cfg, target)
        print("✅ Database upgraded successfully")
        
        # Show current revision
        command.current(alembic_cfg)
        
    except Exception as e:
        print(f"❌ Failed to upgrade database: {e}")
        return False
    
    return True

def downgrade_database(target):
    """Downgrade database to target revision"""
    print(f"⬇️  Downgrading database to {target}...")
    
    alembic_cfg = get_alembic_config()
    
    try:
        command.downgrade(alembic_cfg, target)
        print("✅ Database downgraded successfully")
        
        # Show current revision
        command.current(alembic_cfg)
        
    except Exception as e:
        print(f"❌ Failed to downgrade database: {e}")
        return False
    
    return True

def show_history():
    """Show migration history"""
    print("📜 Migration History:")
    
    alembic_cfg = get_alembic_config()
    
    try:
        command.history(alembic_cfg, verbose=True)
    except Exception as e:
        print(f"❌ Failed to show history: {e}")
        return False
    
    return True

def show_current():
    """Show current revision"""
    print("📍 Current Database Revision:")
    
    alembic_cfg = get_alembic_config()
    
    try:
        command.current(alembic_cfg, verbose=True)
    except Exception as e:
        print(f"❌ Failed to show current revision: {e}")
        return False
    
    return True

def check_database():
    """Check database status and connectivity"""
    print("🔍 Checking database status...")
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        print(f"Database URL: {settings.database_url}")
        print(f"Environment: {settings.environment}")
        
        # Try to connect
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.fetchone():
                print("✅ Database connection successful")
        
        # Check if Alembic table exists
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if "alembic_version" in tables:
            print("✅ Alembic version table found")
            show_current()
        else:
            print("⚠️  Alembic version table not found - database may need initialization")
        
        print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    
    return True

def reset_database():
    """Reset database (WARNING: Destructive operation)"""
    print("⚠️  WARNING: This will destroy all data!")
    
    confirmation = input("Type 'RESET' to confirm: ")
    if confirmation != "RESET":
        print("Operation cancelled")
        return False
    
    # Remove SQLite database file if it exists
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/local/annotations.db")
    if database_url.startswith("sqlite"):
        db_path = database_url.replace("sqlite:///", "")
        db_file = project_root / db_path
        if db_file.exists():
            db_file.unlink()
            print(f"🗑️  Removed database file: {db_file}")
    
    # Initialize fresh database
    return init_database()

def main():
    parser = argparse.ArgumentParser(description="Database Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    subparsers.add_parser("init", help="Initialize database")
    
    # Migration commands
    migration_parser = subparsers.add_parser("migrate", help="Create new migration")
    migration_parser.add_argument("message", help="Migration message")
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade database")
    upgrade_parser.add_argument("target", nargs="?", default="head", help="Target revision (default: head)")
    
    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument("target", help="Target revision")
    
    # Info commands
    subparsers.add_parser("history", help="Show migration history")
    subparsers.add_parser("current", help="Show current revision")
    subparsers.add_parser("check", help="Check database status")
    
    # Reset command
    subparsers.add_parser("reset", help="Reset database (WARNING: Destructive)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    success = False
    
    if args.command == "init":
        success = init_database()
    elif args.command == "migrate":
        success = create_migration(args.message)
    elif args.command == "upgrade":
        success = upgrade_database(args.target)
    elif args.command == "downgrade":
        success = downgrade_database(args.target)
    elif args.command == "history":
        success = show_history()
    elif args.command == "current":
        success = show_current()
    elif args.command == "check":
        success = check_database()
    elif args.command == "reset":
        success = reset_database()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
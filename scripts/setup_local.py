#!/usr/bin/env python3
"""
Local Development Setup Script
Initializes everything needed for completely offline operation
"""

import os
import sys
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
import logging
import hashlib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LocalSetup:
    """Setup local development environment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "local"
        self.config_file = self.project_root / "config" / "local_config.yaml"
        
    def check_python_version(self):
        """Ensure Python 3.8+"""
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required. You have Python %s", sys.version)
            sys.exit(1)
        logger.info("✓ Python version: %s", sys.version.split()[0])
        
    def create_directories(self):
        """Create all required directories"""
        dirs = [
            self.data_dir,
            self.data_dir / "documents",
            self.data_dir / "candidates", 
            self.data_dir / "gold",
            self.data_dir / "exports",
            self.data_dir / "logs",
            self.data_dir / "queue",
            self.data_dir / "llm_cache",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info("✓ Created local data directories")
        
    def setup_sqlite_database(self):
        """Initialize SQLite database with schema"""
        db_path = self.data_dir / "annotations.db"
        
        # Create database and tables
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables
        sql_schema = """
        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            pub_date TEXT,
            raw_text TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Sentences table
        CREATE TABLE IF NOT EXISTS sentences (
            sent_id TEXT,
            doc_id TEXT,
            start_offset INTEGER,
            end_offset INTEGER,
            text TEXT NOT NULL,
            paragraph_id INTEGER,
            metadata TEXT,
            PRIMARY KEY (doc_id, sent_id),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );
        
        -- Candidates table (LLM/Rule suggestions)
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            sent_id TEXT,
            source TEXT CHECK(source IN ('llm', 'rule', 'manual')),
            entity_data TEXT,
            relation_data TEXT, 
            topic_data TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );
        
        -- Gold Annotations table
        CREATE TABLE IF NOT EXISTS gold_annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            sent_id TEXT,
            candidate_id INTEGER,
            annotation_type TEXT CHECK(annotation_type IN ('entity', 'relation', 'topic')),
            annotation_data TEXT NOT NULL,
            annotator TEXT,
            confidence REAL,
            decision TEXT CHECK(decision IN ('accept', 'reject', 'modify')),
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );
        
        -- Triage Queue table
        CREATE TABLE IF NOT EXISTS triage_queue (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            sent_id TEXT,
            candidate_id INTEGER,
            priority_score REAL,
            priority_level TEXT,
            status TEXT DEFAULT 'pending',
            assigned_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );
        
        -- Users table (simple auth)
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            token TEXT UNIQUE NOT NULL,
            role TEXT CHECK(role IN ('admin', 'annotator', 'reviewer')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Annotation Stats table
        CREATE TABLE IF NOT EXISTS annotation_stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            annotations_count INTEGER DEFAULT 0,
            accept_count INTEGER DEFAULT 0,
            reject_count INTEGER DEFAULT 0,
            modify_count INTEGER DEFAULT 0,
            avg_confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_candidates_doc ON candidates(doc_id, sent_id);
        CREATE INDEX IF NOT EXISTS idx_gold_doc ON gold_annotations(doc_id, sent_id);
        CREATE INDEX IF NOT EXISTS idx_queue_status ON triage_queue(status, priority_score);
        CREATE INDEX IF NOT EXISTS idx_queue_assigned ON triage_queue(assigned_to, status);
        """
        
        # Execute schema
        for statement in sql_schema.split(';'):
            if statement.strip():
                cursor.execute(statement)
        
        conn.commit()
        conn.close()
        
        logger.info("✓ SQLite database initialized")
        
    def create_default_users(self):
        """Add default users for local testing"""
        db_path = self.data_dir / "annotations.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        users = [
            ('admin', 'local-admin-2024', 'admin'),
            ('annotator1', 'anno-team-001', 'annotator'),
            ('annotator2', 'anno-team-002', 'annotator'),
            ('reviewer', 'review-lead-003', 'reviewer'),
        ]
        
        for username, token, role in users:
            cursor.execute("""
                INSERT OR IGNORE INTO users (username, token, role) 
                VALUES (?, ?, ?)
            """, (username, token, role))
        
        conn.commit()
        conn.close()
        
        logger.info("✓ Created default users")
        
    def add_sample_data(self):
        """Add sample documents for testing"""
        db_path = self.data_dir / "annotations.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Sample document
        sample_text = """Penaeus vannamei cultured at 28°C showed signs of AHPND infection. 
        The shrimp were treated with oxytetracycline at 50 mg/kg feed for 7 days.
        Vibrio parahaemolyticus was isolated from affected shrimp with 80% mortality rate.
        Water quality parameters: pH 7.8, salinity 25 ppt, dissolved oxygen 5.2 mg/L."""
        
        doc_id = hashlib.md5(sample_text.encode()).hexdigest()[:12]
        
        # Insert document
        cursor.execute("""
            INSERT OR IGNORE INTO documents (doc_id, source, title, raw_text, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, 'sample', 'Sample Shrimp Disease Report', sample_text, '{}'))
        
        # Insert sentences
        sentences = sample_text.split('. ')
        for i, sent in enumerate(sentences):
            if sent.strip():
                cursor.execute("""
                    INSERT OR IGNORE INTO sentences (sent_id, doc_id, start_offset, end_offset, text)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"s{i}", doc_id, i*100, (i+1)*100, sent.strip() + '.'))
        
        conn.commit()
        conn.close()
        
        logger.info("✓ Added sample data for testing")
        
    def setup_virtual_environment(self):
        """Check or create Python virtual environment"""
        venv_path = self.project_root / "venv"
        
        if not venv_path.exists():
            logger.info("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            logger.info("✓ Virtual environment created")
        else:
            logger.info("✓ Virtual environment exists")
            
        # Path to pip in venv
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip"
            python_path = venv_path / "Scripts" / "python"
        else:  # Unix/MacOS
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"
            
        return str(python_path), str(pip_path)
        
    def install_dependencies(self, pip_path):
        """Install required Python packages"""
        # Use local requirements for lighter install
        local_req = self.project_root / "requirements-local.txt"
        main_req = self.project_root / "requirements.txt"
        
        requirements_file = local_req if local_req.exists() else main_req
        
        if requirements_file.exists():
            logger.info(f"Installing Python dependencies from {requirements_file.name}...")
            # Install with specific versions for compatibility
            subprocess.run([
                pip_path, "install", 
                "-r", str(requirements_file),
                "--upgrade"
            ], check=True)
            
            logger.info("✓ Python dependencies installed")
        else:
            logger.warning("requirements file not found")
            
    def check_ollama(self):
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                logger.info("✓ Ollama is installed")
                
                # Check for models
                if "llama" not in result.stdout.lower():
                    logger.info("  Downloading llama3.2:3b model (this may take a few minutes)...")
                    subprocess.run(["ollama", "pull", "llama3.2:3b"], check=True)
                    logger.info("  ✓ Model downloaded")
                else:
                    logger.info("  ✓ Ollama models available")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("⚠ Ollama not found - will use rule-based annotation only")
            logger.info("  To enable LLM: Install from https://ollama.ai")
            return False
            
    def setup_frontend(self):
        """Check and setup frontend dependencies"""
        ui_path = self.project_root / "ui"
        
        if not (ui_path / "node_modules").exists():
            logger.info("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=str(ui_path), check=True)
            logger.info("✓ Frontend dependencies installed")
        else:
            logger.info("✓ Frontend dependencies exist")
            
    def create_env_file(self, has_ollama):
        """Create .env file for local configuration"""
        env_path = self.project_root / ".env.local"
        
        env_content = f"""# Local Development Environment Variables
# Generated on {datetime.now().isoformat()}

# Database
DATABASE_URL=sqlite:///./data/local/annotations.db

# LLM Configuration
LLM_PROVIDER={'ollama' if has_ollama else 'rules_only'}
LLM_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434

# Cache
CACHE_TYPE=memory
CACHE_DIR=./data/local/cache

# API Configuration  
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true

# UI Configuration
REACT_APP_API_URL=http://localhost:8000

# Authentication
AUTH_ENABLED=true
AUTH_TYPE=simple_token

# Paths
DATA_DIR=./data/local
LOG_DIR=./data/local/logs

# Development
DEBUG=true
ENVIRONMENT=local
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
            
        logger.info("✓ Created .env.local file")
        
    def create_start_script(self):
        """Create convenient start script"""
        script_path = self.project_root / "start_local.sh"
        
        script_content = """#!/bin/bash
# Start Local Development Environment

echo "🚀 Starting Shrimp Annotation Pipeline (Local Mode)"
echo "================================================"

# Activate virtual environment
source venv/bin/activate

# Export environment variables
export $(grep -v '^#' .env.local | xargs)

# Check if Ollama is running (optional)
if command -v ollama &> /dev/null; then
    echo "✓ Ollama detected"
    # Ensure Ollama is serving
    ollama serve > /dev/null 2>&1 &
    sleep 2
else
    echo "⚠ Ollama not found - using rules-only mode"
fi

# Start Backend API
echo "Starting API server..."
python services/api/local_annotation_api.py &
API_PID=$!
echo "✓ API server started (PID: $API_PID)"

# Wait for API to be ready
sleep 3

# Start Frontend
echo "Starting UI server..."
cd ui && npm start &
UI_PID=$!
echo "✓ UI server started (PID: $UI_PID)"

echo ""
echo "================================================"
echo "✅ Local environment is running!"
echo ""
echo "Access points:"
echo "  📊 UI:       http://localhost:3000"
echo "  🔌 API:      http://localhost:8000"
echo "  📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Default login tokens:"
echo "  Admin:      local-admin-2024"
echo "  Annotator:  anno-team-001"
echo "  Reviewer:   review-lead-003"
echo ""
echo "Press Ctrl+C to stop all services"
echo "================================================"

# Wait for interrupt
trap "kill $API_PID $UI_PID 2>/dev/null; echo 'Services stopped'" EXIT
wait
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info("✓ Created start_local.sh script")
        
    def print_summary(self):
        """Print setup summary and instructions"""
        print("\n" + "="*60)
        print("✅ LOCAL SETUP COMPLETE!")
        print("="*60)
        print("\n📋 Quick Start Guide:")
        print("-"*40)
        print("1. Activate environment:")
        print("   source venv/bin/activate")
        print("\n2. Start everything:")
        print("   ./start_local.sh")
        print("\n3. Access the application:")
        print("   🌐 http://localhost:3000")
        print("\n4. Login with token:")
        print("   local-admin-2024")
        print("\n" + "="*60)
        print("\n💡 Tips:")
        print("  • All data stored in ./data/local/")
        print("  • SQLite database at ./data/local/annotations.db")
        print("  • No internet connection required!")
        print("  • To reset: rm -rf ./data/local/")
        print("="*60 + "\n")
        
    def run(self):
        """Run complete setup"""
        print("\n🔧 Setting up Local Development Environment\n")
        
        try:
            # 1. Check Python version
            self.check_python_version()
            
            # 2. Create directories
            self.create_directories()
            
            # 3. Setup database
            self.setup_sqlite_database()
            
            # 4. Create users
            self.create_default_users()
            
            # 5. Add sample data
            self.add_sample_data()
            
            # 6. Setup virtual environment
            python_path, pip_path = self.setup_virtual_environment()
            
            # 7. Install dependencies
            self.install_dependencies(pip_path)
            
            # 8. Check Ollama
            has_ollama = self.check_ollama()
            
            # 9. Setup frontend
            self.setup_frontend()
            
            # 10. Create env file
            self.create_env_file(has_ollama)
            
            # 11. Create start script
            self.create_start_script()
            
            # 12. Print summary
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    setup = LocalSetup()
    setup.run()
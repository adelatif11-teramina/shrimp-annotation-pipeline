#!/usr/bin/env python3
"""
Setup script for Shrimp Annotation Pipeline

Initializes the project, sets up databases, and configures services.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Setup and configuration manager"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_training_path = project_root.parent / "data-training"
        
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        logger.info("Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
        
        # Check Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            logger.info("✓ Docker installed")
        except:
            logger.error("Docker not found. Please install Docker.")
            return False
        
        # Check data-training project
        if self.data_training_path.exists():
            logger.info(f"✓ Found data-training project at {self.data_training_path}")
        else:
            logger.warning(f"Data-training project not found at {self.data_training_path}")
        
        return True
    
    def setup_virtual_env(self):
        """Setup Python virtual environment"""
        logger.info("Setting up virtual environment...")
        
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            logger.info("✓ Created virtual environment")
        
        # Install dependencies
        pip_path = venv_path / "bin" / "pip" if os.name != "nt" else venv_path / "Scripts" / "pip.exe"
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        logger.info("✓ Installed dependencies")
        
        # Download spaCy model
        python_path = venv_path / "bin" / "python" if os.name != "nt" else venv_path / "Scripts" / "python.exe"
        try:
            subprocess.run([str(python_path), "-m", "spacy", "download", "en_core_web_sm"], check=True)
            logger.info("✓ Downloaded spaCy model")
        except:
            logger.warning("Failed to download spaCy model (optional)")
    
    def setup_configuration(self):
        """Create configuration files"""
        logger.info("Setting up configuration...")
        
        # Create .env file
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = """# Environment Configuration
# OPENAI_API_KEY=sk-your-key-here
# OLLAMA_HOST=http://localhost:11434

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=annotations
DB_USER=annotator
DB_PASSWORD=secure_password_change_me

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Label Studio
LABEL_STUDIO_URL=http://localhost:8080
# LABEL_STUDIO_TOKEN=your-token-here

# Data paths
DATA_TRAINING_PATH=../data-training
GOLD_STORE_PATH=./data/gold
"""
            env_file.write_text(env_content)
            logger.info("✓ Created .env file")
        
        # Create main config
        config_file = self.project_root / "config" / "settings.yaml"
        if not config_file.exists():
            config_file.parent.mkdir(exist_ok=True)
            config_content = """# Annotation Pipeline Configuration

# Triage prioritization weights
triage:
  weights:
    confidence: 0.2
    novelty: 0.3
    impact: 0.25
    disagreement: 0.15
    authority: 0.1
  thresholds:
    critical: 0.9
    high: 0.7
    medium: 0.5
    low: 0.3

# Source authority scores
source_authority:
  paper: 1.0
  report: 0.9
  manual: 0.8
  hatchery_log: 0.6
  dataset: 0.7

# LLM settings
llm:
  provider: openai  # or ollama
  model: gpt-5
  temperature: 0.1
  max_tokens: 500
  cache_enabled: true

# Annotation settings
annotation:
  batch_size: 10
  auto_accept_threshold: 0.95
  require_double_annotation: true
  iaa_sample_rate: 0.2
"""
            config_file.write_text(config_content)
            logger.info("✓ Created settings.yaml")
    
    def import_sample_data(self):
        """Import sample data from data-training project"""
        logger.info("Importing sample data...")
        
        source_text = self.data_training_path / "data/output/text"
        dest_raw = self.project_root / "data/raw"
        
        if source_text.exists():
            # Copy a few sample files
            sample_files = list(source_text.glob("*.txt"))[:3]
            for file in sample_files:
                dest_file = dest_raw / file.name
                if not dest_file.exists():
                    shutil.copy(file, dest_file)
                    logger.info(f"  Imported {file.name}")
        
        # Also import existing annotations if available
        source_annotations = self.data_training_path / "data/annotation"
        if source_annotations.exists():
            sample_jsonl = list(source_annotations.glob("*.jsonl"))[:1]
            for file in sample_jsonl:
                dest_file = dest_raw / file.name
                if not dest_file.exists():
                    shutil.copy(file, dest_file)
                    logger.info(f"  Imported {file.name}")
    
    def setup_label_studio_project(self):
        """Create Label Studio project configuration"""
        logger.info("Setting up Label Studio configuration...")
        
        ls_config = self.project_root / "config" / "label_studio_project.json"
        
        # Read the Label Studio config XML from data-training
        ls_xml_source = self.data_training_path / "config/label_studio_config.xml"
        if ls_xml_source.exists():
            with open(ls_xml_source, 'r') as f:
                label_config = f.read()
        else:
            # Use default config
            label_config = """<View>
  <Text name="text" value="$text"/>
  <Labels name="label" toName="text">
    <Label value="SPECIES" background="#FF6B6B"/>
    <Label value="PATHOGEN" background="#FF8E53"/>
    <Label value="DISEASE" background="#FFD93D"/>
    <Label value="TREATMENT" background="#6BCF7F"/>
  </Labels>
</View>"""
        
        project_config = {
            "title": "Shrimp Aquaculture Annotation",
            "description": "HITL annotation for shrimp farming knowledge graph",
            "label_config": label_config,
            "expert_instruction": "Please follow the annotation guidelines carefully.",
            "show_instruction": True,
            "show_skip_button": True,
            "enable_empty_annotation": False,
            "show_annotation_history": True,
            "maximum_annotations": 2,
            "min_annotations_to_start_training": 50
        }
        
        ls_config.write_text(json.dumps(project_config, indent=2))
        logger.info("✓ Created Label Studio project config")
    
    def start_services(self, detached: bool = True):
        """Start Docker services"""
        logger.info("Starting Docker services...")
        
        cmd = ["docker-compose", "up"]
        if detached:
            cmd.append("-d")
        
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
            logger.info("✓ Started services")
            
            if detached:
                logger.info("\nServices running at:")
                logger.info("  Label Studio: http://localhost:8080")
                logger.info("  PostgreSQL: localhost:5432")
                logger.info("  Redis: localhost:6379")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start services: {e}")
    
    def run_full_setup(self):
        """Run complete setup"""
        logger.info("=== Shrimp Annotation Pipeline Setup ===\n")
        
        if not self.check_prerequisites():
            return False
        
        self.setup_virtual_env()
        self.setup_configuration()
        self.import_sample_data()
        self.setup_label_studio_project()
        
        logger.info("\n=== Setup Complete ===")
        logger.info("\nNext steps:")
        logger.info("1. Edit .env file with your API keys")
        logger.info("2. Run: docker-compose up -d")
        logger.info("3. Access Label Studio at http://localhost:8080")
        logger.info("4. Create admin account and import project config")
        logger.info("5. Start annotating!\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Setup Shrimp Annotation Pipeline")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--start-services", action="store_true", help="Start Docker services")
    parser.add_argument("--import-data", action="store_true", help="Import sample data only")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    setup = ProjectSetup(project_root)
    
    if args.import_data:
        setup.import_sample_data()
    elif args.start_services:
        setup.start_services()
    else:
        setup.run_full_setup()


if __name__ == "__main__":
    main()
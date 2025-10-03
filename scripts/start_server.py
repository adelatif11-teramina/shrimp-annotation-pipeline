#!/usr/bin/env python3
"""
Production server startup script
Handles database initialization, migrations, and server startup
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    required_vars = ["JWT_SECRET_KEY", "DB_PASSWORD"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or environment configuration")
        return False
    
    # Check for production-specific requirements
    if os.getenv("ENVIRONMENT") == "production":
        if os.getenv("JWT_SECRET_KEY") == "development_secret_key_change_in_production":
            print("❌ JWT_SECRET_KEY must be changed from default in production")
            return False
        
        if os.getenv("DB_PASSWORD") == "dev_password_change_in_production":
            print("❌ DB_PASSWORD must be changed from default in production")
            return False
    
    print("✅ Environment configuration looks good")
    return True

def initialize_database():
    """Initialize or upgrade database"""
    print("🏗️  Initializing database...")
    
    try:
        # Run database management script
        result = subprocess.run([
            sys.executable, 
            str(project_root / "scripts/manage_db.py"), 
            "check"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Database check failed, attempting initialization...")
            
            # Try to initialize
            result = subprocess.run([
                sys.executable,
                str(project_root / "scripts/manage_db.py"),
                "init"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Database initialization failed: {result.stderr}")
                return False
        
        print("✅ Database is ready")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API server...")
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        # Import the FastAPI app
        from services.api.annotation_api import app
        
        # Start with uvicorn
        import uvicorn
        
        uvicorn.run(
            "services.api.annotation_api:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.is_development,
            workers=1 if settings.is_development else settings.api_workers,
            log_level=settings.log_level,
            access_log=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def print_banner():
    """Print startup banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Shrimp Annotation Pipeline API                 ║
║                   Production Server                         ║
╚══════════════════════════════════════════════════════════════╝
""")

def print_server_info():
    """Print server information"""
    from config.settings import get_settings
    settings = get_settings()
    
    print("\n📊 Server Configuration:")
    print(f"   Environment: {settings.environment}")
    print(f"   Host: {settings.api_host}")
    print(f"   Port: {settings.api_port}")
    print(f"   Workers: {1 if settings.is_development else settings.api_workers}")
    print(f"   Debug: {settings.debug}")
    print(f"   Database: {settings.database_url.split('://', 1)[0]}://***")
    print(f"   Rate Limiting: {'Enabled' if settings.rate_limit_enabled else 'Disabled'}")
    
    if settings.openai_api_key:
        print(f"   OpenAI: Configured")
    else:
        print(f"   OpenAI: Not configured")
    
    print("\n🔗 API Endpoints:")
    print(f"   Health Check: http://{settings.api_host}:{settings.api_port}/health")
    print(f"   API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"   Authentication: http://{settings.api_host}:{settings.api_port}/auth/login")
    
    if settings.frontend_url:
        print(f"   Frontend: {settings.frontend_url}")
    
    print()

def main():
    """Main startup sequence"""
    print_banner()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Initialize database
    if not initialize_database():
        sys.exit(1)
    
    # Print server info
    print_server_info()
    
    # Start server
    try:
        start_api_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
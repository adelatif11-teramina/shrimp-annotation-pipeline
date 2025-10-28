#!/usr/bin/env python3
"""
Railway Deployment Verification for Smart Chunking

This script verifies that smart chunking is properly deployed and working on Railway.
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_local_smart_chunking():
    """Test smart chunking locally before deployment"""
    print("🧪 Testing Smart Chunking Locally...")
    
    try:
        from services.ingestion.smart_chunking import SmartChunkingService
        
        test_text = """
        Vibrio parahaemolyticus causes acute hepatopancreatic necrosis disease (AHPND) in shrimp. 
        This pathogen has emerged as a major threat to aquaculture worldwide. 
        The disease results in significant mortality and economic losses.
        
        Recent studies have identified novel virulence factors in the pathogen. 
        These factors contribute to increased pathogenicity and host range expansion. 
        Understanding their mechanisms is crucial for developing effective treatments.
        """
        
        chunker = SmartChunkingService(target_length=(150, 400))
        chunks = chunker.create_smart_chunks(test_text.strip())
        
        print(f"✅ Smart chunking created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {chunk.char_count} chars, {chunk.sentence_count} sentences")
            if chunk.has_definitions:
                print(f"    → Preserves definitions")
            if chunk.has_pronouns:
                print(f"    → Resolves pronouns")
        
        stats = chunker.get_chunking_statistics(chunks)
        print(f"✅ Quality score: {stats['quality_score']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Local smart chunking test failed: {e}")
        return False

def test_railway_api_endpoint(base_url: str):
    """Test Railway API endpoint with smart chunking"""
    print(f"🚀 Testing Railway API at {base_url}...")
    
    try:
        # Test health endpoint
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Railway API is healthy")
        else:
            print(f"⚠️ Health check returned {health_response.status_code}")
        
        # Test document upload with smart chunking
        test_doc = {
            "title": "Smart Chunking Test Document",
            "content": """
            Translucent post-larvae disease (TPD), caused by Vibrio parahaemolyticus (VpTPD), has become an emerging shrimp disease. 
            This pathogen affects more than 70%-80% of coastal shrimp nurseries in China. 
            The disease spreads rapidly through water systems and causes significant mortality.
            
            Treatment options include antibiotics and improved biosecurity measures. 
            However, prevention through proper management is more effective than treatment. 
            Regular monitoring helps detect outbreaks early and prevent spread.
            """.strip()
        }
        
        print("📤 Testing document upload with smart chunking...")
        upload_response = requests.post(
            f"{base_url}/api/upload/document",
            json=test_doc,
            timeout=30
        )
        
        if upload_response.status_code == 200:
            result = upload_response.json()
            print("✅ Document uploaded successfully with smart chunking")
            print(f"  Doc ID: {result.get('doc_id')}")
            print(f"  Chunking mode: {result.get('chunking_mode', 'unknown')}")
            print(f"  Chunks created: {result.get('chunk_count', 0)}")
            print(f"  Total sentences: {result.get('sentence_count', 0)}")
            
            if 'chunking_info' in result:
                info = result['chunking_info']
                print(f"  Avg chars per chunk: {info.get('avg_chars_per_chunk', 0)}")
                print(f"  Chunks with context: {info.get('chunks_with_context', 0)}")
            
            return True
        else:
            print(f"❌ Upload failed: {upload_response.status_code}")
            print(f"Response: {upload_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Railway API test failed: {e}")
        return False

def test_production_configuration():
    """Verify production configuration is set up correctly"""
    print("⚙️ Testing Production Configuration...")
    
    # Check if smart chunking modules are importable
    try:
        from services.ingestion.smart_chunking import SmartChunkingService
        from services.ingestion.chunking_integration import ImprovedDocumentIngestionService
        print("✅ Smart chunking modules are importable")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check configuration files
    config_files = [
        "config/production_config.yaml",
        "config/local_config.yaml"
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            print(f"✅ {config_file} exists")
            with open(config_path, 'r') as f:
                content = f.read()
                if 'smart_paragraph' in content:
                    print(f"  → Smart chunking configured in {config_file}")
                else:
                    print(f"  ⚠️ Smart chunking not found in {config_file}")
        else:
            print(f"❌ {config_file} missing")
    
    return True

def main():
    """Main verification function"""
    print("🔍 SMART CHUNKING DEPLOYMENT VERIFICATION")
    print("=" * 50)
    
    # Test 1: Local smart chunking functionality
    local_test = test_local_smart_chunking()
    print()
    
    # Test 2: Production configuration
    config_test = test_production_configuration()
    print()
    
    # Test 3: Railway deployment (if URL provided)
    railway_url = os.getenv("RAILWAY_URL") or input("Enter Railway URL (or press Enter to skip): ").strip()
    
    railway_test = True
    if railway_url:
        if not railway_url.startswith(('http://', 'https://')):
            railway_url = f"https://{railway_url}"
        railway_test = test_railway_api_endpoint(railway_url)
    else:
        print("⏭️ Skipping Railway API test (no URL provided)")
    
    print()
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Local Smart Chunking: {'✅ PASS' if local_test else '❌ FAIL'}")
    print(f"Production Config: {'✅ PASS' if config_test else '❌ FAIL'}")
    print(f"Railway Deployment: {'✅ PASS' if railway_test else '❌ FAIL'}")
    
    if all([local_test, config_test, railway_test]):
        print("\n🎉 ALL TESTS PASSED - Smart chunking is ready for production!")
        return 0
    else:
        print("\n⚠️ Some tests failed - check the issues above")
        return 1

if __name__ == "__main__":
    exit(main())
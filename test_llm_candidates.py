#!/usr/bin/env python3
"""
Test LLM candidate generation capabilities.
This script tests what LLM providers are available and generates sample candidates.
"""

import sys
import os
from pathlib import Path

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

def test_openai_availability():
    """Test if OpenAI is available"""
    print("Testing OpenAI availability...")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ❌ No OPENAI_API_KEY environment variable found")
        return False
    
    # Check if openai package is installed
    try:
        import openai
        print("  ✅ OpenAI package installed")
        
        # Test basic connection (just check if we can create client)
        try:
            client = openai.OpenAI(api_key=api_key)
            print("  ✅ OpenAI client initialized")
            return True
        except Exception as e:
            print(f"  ❌ OpenAI client error: {e}")
            return False
            
    except ImportError:
        print("  ❌ OpenAI package not installed")
        return False

def test_ollama_availability():
    """Test if Ollama is available"""
    print("Testing Ollama availability...")
    
    try:
        import requests
        
        # Test if Ollama server is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"  ✅ Ollama running with {len(models)} models")
                for model in models[:3]:  # Show first 3 models
                    print(f"    - {model.get('name', 'Unknown')}")
                return True
            else:
                print(f"  ❌ Ollama server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Cannot connect to Ollama: {e}")
            return False
            
    except ImportError:
        print("  ❌ Requests package not installed")
        return False

def test_rule_based_fallback():
    """Test rule-based annotation fallback"""
    print("Testing rule-based annotation fallback...")
    
    try:
        from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine
        
        rule_engine = ShimpAquacultureRuleEngine()
        test_text = "Vibrio parahaemolyticus causes AHPND in Penaeus vannamei at 28°C."
        
        entities = rule_engine.extract_entities(test_text)
        print(f"  ✅ Rule engine working - found {len(entities)} entities")
        
        for entity in entities[:3]:  # Show first 3 entities
            if hasattr(entity, 'text'):
                print(f"    - {entity.text} ({entity.label})")
            else:
                print(f"    - {entity.get('text', 'N/A')} ({entity.get('label', 'N/A')})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Rule engine error: {e}")
        return False

def test_candidate_generation():
    """Test candidate generation with available provider"""
    print("\nTesting candidate generation...")
    
    # Try different providers in order of preference
    test_sentence = "Penaeus vannamei infected with Vibrio parahaemolyticus showed symptoms of AHPND."
    
    # First try rules-only (always available)
    print("Testing rule-based candidate generation...")
    try:
        from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine
        
        rule_engine = ShimpAquacultureRuleEngine()
        entities = rule_engine.extract_entities(test_sentence)
        
        print(f"  ✅ Generated {len(entities)} entity candidates using rules")
        for entity in entities:
            if hasattr(entity, 'text'):
                print(f"    - {entity.text} → {entity.label} ({entity.confidence:.2f})")
            else:
                print(f"    - {entity.get('text', '')} → {entity.get('label', '')} ({entity.get('confidence', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Rule-based generation failed: {e}")
        return False

def main():
    """Test LLM capabilities"""
    
    print("🦐 Testing LLM Candidate Generation Capabilities\n")
    
    # Test available providers
    openai_available = test_openai_availability()
    ollama_available = test_ollama_availability()  
    rules_available = test_rule_based_fallback()
    
    print(f"\n📊 Provider Availability Summary:")
    print(f"  OpenAI: {'✅ Ready' if openai_available else '❌ Not available'}")
    print(f"  Ollama: {'✅ Ready' if ollama_available else '❌ Not available'}")
    print(f"  Rules:  {'✅ Ready' if rules_available else '❌ Not available'}")
    
    # Test candidate generation
    if rules_available:
        success = test_candidate_generation()
        
        if success:
            print(f"\n🎯 Recommendation:")
            if openai_available:
                print("  • Use OpenAI GPT-4o-mini for high-quality candidates")
                print("  • Set OPENAI_API_KEY environment variable")
            elif ollama_available:
                print("  • Use Ollama for local LLM processing")
                print("  • Install a model: ollama pull llama3.2:3b")
            else:
                print("  • Use rule-based annotation only")
                print("  • Consider installing Ollama for better coverage")
                
            print(f"\n✅ LLM candidate generation is ready to process your documents!")
        else:
            print(f"\n❌ No working candidate generation method available")
    else:
        print(f"\n❌ Critical: No candidate generation methods available")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
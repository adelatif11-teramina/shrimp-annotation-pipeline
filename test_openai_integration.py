#!/usr/bin/env python3
"""
Test OpenAI integration for high-quality annotation candidates.
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

def test_openai_connection():
    """Test basic OpenAI connection"""
    print("🔑 Testing OpenAI Connection...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ❌ No OPENAI_API_KEY found in environment")
        return False
        
    print(f"  ✅ API key found: {api_key[:20]}...")
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        print("  ✅ OpenAI client initialized")
        return True
    except ImportError:
        print("  ❌ OpenAI package not installed")
        return False
    except Exception as e:
        print(f"  ❌ OpenAI client error: {e}")
        return False

def test_llm_candidate_generator():
    """Test LLM candidate generator with OpenAI"""
    print("\n🤖 Testing LLM Candidate Generator...")
    
    try:
        from services.candidates.llm_candidate_generator import LLMCandidateGenerator
        
        # Initialize with OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        generator = LLMCandidateGenerator(
            provider="openai",
            model="gpt-4o",
            api_key=api_key,
            temperature=0.1,
            cache_dir=pipeline_root / "data/local/llm_cache"
        )
        
        print("  ✅ LLM generator initialized with OpenAI")
        return generator
        
    except Exception as e:
        print(f"  ❌ Error initializing LLM generator: {e}")
        return None

async def test_entity_extraction(generator):
    """Test entity extraction with real sentences"""
    print("\n🦐 Testing Entity Extraction...")
    
    test_sentences = [
        "Vibrio parahaemolyticus causes AHPND in Penaeus vannamei post-larvae at 28°C.",
        "Treatment with florfenicol at 10 mg/kg improved survival rate in infected shrimp.",
        "The PvIGF gene was associated with growth rate and disease resistance in SPR lines.",
        "Post-larvae stocked at 15 PL/m² showed better FCR than higher densities.",
        "White spot syndrome virus (WSSV) infection resulted in 80% mortality."
    ]
    
    print(f"  Testing {len(test_sentences)} sentences...")
    
    results = []
    for i, sentence in enumerate(test_sentences):
        try:
            print(f"\n  Sentence {i+1}: \"{sentence[:50]}...\"")
            
            # Extract entities (async call)
            entities = await generator.extract_entities(sentence)
            
            print(f"    ✅ Found {len(entities)} entities:")
            for entity in entities[:5]:  # Show first 5
                print(f"      - {entity.text} → {entity.label} ({entity.confidence:.2f})")
            
            results.append({
                "sentence": sentence,
                "entities": [
                    {
                        "text": e.text,
                        "label": e.label,
                        "start": e.start,
                        "end": e.end,
                        "confidence": e.confidence
                    }
                    for e in entities
                ]
            })
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    return results

def compare_with_rules(test_results):
    """Compare OpenAI results with rule-based results"""
    print("\n📊 Comparing OpenAI vs Rules...")
    
    try:
        from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine
        rule_engine = ShimpAquacultureRuleEngine()
        
        for result in test_results[:2]:  # Compare first 2 sentences
            sentence = result["sentence"]
            openai_entities = result["entities"]
            
            # Get rule-based entities
            rule_entities = rule_engine.extract_entities(sentence)
            
            print(f"\n  Sentence: \"{sentence[:50]}...\"")
            print(f"    OpenAI: {len(openai_entities)} entities")
            for e in openai_entities[:3]:
                print(f"      - {e['text']} → {e['label']} ({e['confidence']:.2f})")
            
            print(f"    Rules:  {len(rule_entities)} entities")
            for e in rule_entities[:3]:
                if hasattr(e, 'text'):
                    print(f"      - {e.text} → {e.label} ({e.confidence:.2f})")
        
    except Exception as e:
        print(f"  ❌ Error comparing with rules: {e}")

def save_test_results(results):
    """Save test results for inspection"""
    output_file = pipeline_root / "data/openai_test_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved to: {output_file}")

async def main():
    """Test OpenAI integration"""
    
    print("🚀 OpenAI Integration Test for Shrimp Annotation Pipeline\n")
    
    # Test connection
    if not test_openai_connection():
        print("\n❌ OpenAI connection failed. Please check your API key.")
        return 1
    
    # Test LLM generator
    generator = test_llm_candidate_generator()
    if not generator:
        print("\n❌ LLM generator initialization failed.")
        return 1
    
    # Test entity extraction
    results = await test_entity_extraction(generator)
    if not results:
        print("\n❌ Entity extraction failed.")
        return 1
    
    # Compare with rules
    compare_with_rules(results)
    
    # Save results
    save_test_results(results)
    
    print(f"\n✅ OpenAI Integration Test Complete!")
    print(f"🎯 Ready to generate high-quality candidates for {len(os.listdir(pipeline_root / 'data/raw'))} documents")
    
    return 0

if __name__ == "__main__":
    # Load environment variables from .env file if available
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API key in environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set it in your .env file or environment:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
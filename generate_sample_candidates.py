#!/usr/bin/env python3
"""
Generate sample annotation candidates from imported documents.
This script demonstrates the candidate generation process using rule-based extraction.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine
from services.ingestion.document_ingestion import DocumentIngestionService

def main():
    """Generate candidates for sample documents"""
    
    print("🦐 Generating Annotation Candidates from Imported Documents\n")
    
    # Initialize services
    print("Initializing services...")
    ingestion_service = DocumentIngestionService(segmenter="regex")
    rule_engine = ShimpAquacultureRuleEngine()
    
    # Get a sample document
    raw_dir = pipeline_root / "data/raw"
    sample_files = ["test1.txt", "noaa.txt", "test2.txt"]
    
    candidates_dir = pipeline_root / "data/candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    total_candidates = 0
    
    for filename in sample_files:
        doc_file = raw_dir / filename
        if not doc_file.exists():
            print(f"⚠️  Skipping {filename} - file not found")
            continue
            
        print(f"\n📄 Processing: {filename}")
        
        try:
            # Ingest document
            document = ingestion_service.ingest_text_file(
                file_path=doc_file,
                source="data-training",
                title=doc_file.stem
            )
            
            print(f"   Segmented into {len(document.sentences)} sentences")
            
            # Generate candidates for first 10 sentences
            doc_candidates = []
            
            for i, sentence in enumerate(document.sentences[:10]):
                try:
                    # Extract entities using rules
                    entities = rule_engine.extract_entities(sentence.text)
                    
                    if entities:
                        candidate = {
                            "doc_id": document.doc_id,
                            "sent_id": sentence.sent_id,
                            "text": sentence.text,
                            "start": sentence.start,
                            "end": sentence.end,
                            "entities": [
                                {
                                    "text": entity.text,
                                    "label": entity.label,
                                    "start": entity.start,
                                    "end": entity.end,
                                    "confidence": entity.confidence,
                                    "rule_source": entity.rule_source
                                }
                                for entity in entities
                            ],
                            "relations": [],  # Rule engine currently focuses on entities
                            "topics": [],    # Could add topic classification here
                            "source": "rule_engine",
                            "created_at": datetime.now().isoformat()
                        }
                        
                        doc_candidates.append(candidate)
                        total_candidates += len(entities)
                        
                except Exception as e:
                    print(f"     Error processing sentence {i}: {e}")
            
            # Save candidates
            if doc_candidates:
                output_file = candidates_dir / f"{document.doc_id}_candidates.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for candidate in doc_candidates:
                        f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
                
                print(f"   ✅ Generated {len(doc_candidates)} sentence candidates")
                print(f"   💾 Saved to: {output_file}")
                
                # Show sample candidate
                if doc_candidates:
                    sample = doc_candidates[0]
                    print(f"   📝 Sample: \"{sample['text'][:80]}...\"")
                    print(f"      Entities: {len(sample['entities'])}")
                    for entity in sample['entities'][:3]:
                        print(f"        - {entity['text']} → {entity['label']} ({entity['confidence']:.2f})")
            else:
                print(f"   ⚠️  No candidates generated")
                
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    print(f"\n🎯 Summary:")
    print(f"   Total entity candidates: {total_candidates}")
    print(f"   Output directory: {candidates_dir}")
    
    # Check if API can access these candidates
    print(f"\n🔗 Next steps:")
    print(f"   1. Candidates are ready for human annotation")
    print(f"   2. Load into annotation pipeline via API")
    print(f"   3. Start annotation workflow")
    
    if total_candidates > 0:
        print(f"\n✅ Rule-based candidate generation is working successfully!")
        print(f"💡 Consider setting up OpenAI API key or Ollama for enhanced candidates")
    else:
        print(f"\n⚠️  No candidates generated - check rule patterns")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Quick one-time data injection: Replace mock data with real document sentences
"""

import json
import random
from pathlib import Path

def extract_real_sentences():
    """Extract sentences from OpenAI candidates"""
    candidates_file = Path("data/candidates/52526b16bfe3_openai_candidates.jsonl")
    
    sentences = []
    if candidates_file.exists():
        with open(candidates_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidate = json.loads(line)
                    entities = candidate.get('entities', [])
                    
                    # Convert to expected format
                    formatted_entities = []
                    for i, ent in enumerate(entities):
                        formatted_entities.append({
                            "id": i + 1,
                            "text": ent.get('text', ''),
                            "label": ent.get('label', 'UNKNOWN'),
                            "start": ent.get('start', 0),
                            "end": ent.get('end', 0),
                            "confidence": ent.get('confidence', 0.8)
                        })
                    
                    sentences.append({
                        "id": candidate.get('candidate_id', len(sentences) + 1),
                        "candidate_id": candidate.get('candidate_id', len(sentences) + 100),
                        "doc_id": candidate.get('doc_id', 'doc1'),
                        "sent_id": candidate.get('sent_id', f's{len(sentences)}'),
                        "text": candidate.get('sentence', candidate.get('text', '')),
                        "priority_score": min(0.95, len(entities) * 0.25 + random.uniform(0.3, 0.7)),
                        "priority_level": "high" if len(entities) > 2 else "medium",
                        "status": "pending",
                        "entities": formatted_entities,
                        "relations": [],
                        "topics": [{"topic": "T_DISEASE", "score": 0.85}] if entities else [],
                        "source": "real_documents"
                    })
    
    return sentences

def main():
    print("💉 Quick Data Injection: Loading Real Document Sentences")
    
    # Extract real sentences
    real_sentences = extract_real_sentences()
    
    if not real_sentences:
        print("❌ No real sentences found")
        return
    
    # Update the existing realistic mock data file
    output_file = Path("services/api/realistic_mock_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(real_sentences, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Injected {len(real_sentences)} real sentences into mock API")
    print("🔄 The mock API will now serve actual document content!")

if __name__ == "__main__":
    main()
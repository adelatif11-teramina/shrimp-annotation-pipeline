#!/usr/bin/env python3
"""
Create realistic mock data for the API based on actual imported documents.
This replaces the generic mock data with real sentences from your documents.
"""

import sys
import json
import random
from pathlib import Path

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

def extract_real_sentences():
    """Extract real sentences from imported documents"""
    print("📖 Extracting real sentences from imported documents...")
    
    raw_dir = pipeline_root / "data/raw"
    real_docs = ["test1.txt", "test2.txt", "noaa.txt"]
    
    sentences = []
    
    for filename in real_docs:
        doc_file = raw_dir / filename
        if not doc_file.exists():
            continue
            
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into sentences and find good ones
            for i, sent in enumerate(content.split('.')):
                sent = sent.strip()
                
                # Filter for good sentences (contain domain terms, reasonable length)
                if (len(sent) > 80 and len(sent) < 200 and
                    any(word in sent.lower() for word in ['shrimp', 'vibrio', 'disease', 'aquaculture', 'larvae', 'pathogen', 'treatment'])):
                    
                    sentences.append({
                        "text": sent + ".",
                        "doc_id": filename.replace('.txt', ''),
                        "source": filename
                    })
                    
                    if len(sentences) >= 20:  # Enough for demo
                        break
            
            if len(sentences) >= 20:
                break
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"   ✅ Extracted {len(sentences)} real sentences")
    return sentences

def create_mock_data_file():
    """Create a file with realistic mock data"""
    
    sentences = extract_real_sentences()
    
    if not sentences:
        print("❌ No sentences extracted")
        return False
    
    # Create realistic triage queue items
    mock_items = []
    
    for i, sent_data in enumerate(sentences):
        # Generate realistic entities based on sentence content
        entities = []
        text = sent_data["text"]
        
        # Find real entities in the text
        if "vibrio" in text.lower():
            start = text.lower().find("vibrio")
            entities.append({
                "id": len(entities) + 1,
                "text": "Vibrio parahaemolyticus" if "parahaemolyticus" in text else "Vibrio",
                "label": "PATHOGEN",
                "start": start,
                "end": start + len("vibrio parahaemolyticus"),
                "confidence": 0.95
            })
        
        if "penaeus" in text.lower() or "shrimp" in text.lower():
            start = text.lower().find("penaeus" if "penaeus" in text.lower() else "shrimp")
            entities.append({
                "id": len(entities) + 1,
                "text": "Penaeus vannamei" if "penaeus" in text.lower() else "shrimp",
                "label": "SPECIES",
                "start": start,
                "end": start + 10,
                "confidence": 0.90
            })
        
        if any(disease in text.lower() for disease in ["ahpnd", "disease", "syndrome"]):
            disease_words = ["ahpnd", "disease", "syndrome"]
            for disease in disease_words:
                if disease in text.lower():
                    start = text.lower().find(disease)
                    entities.append({
                        "id": len(entities) + 1,
                        "text": disease.upper() if disease == "ahpnd" else disease,
                        "label": "DISEASE",
                        "start": start,
                        "end": start + len(disease),
                        "confidence": 0.88
                    })
                    break
        
        # Calculate priority
        priority_score = min(0.95, len(entities) * 0.25 + random.uniform(0.3, 0.7))
        if priority_score > 0.8:
            priority_level = "high"
        elif priority_score > 0.5:
            priority_level = "medium"
        else:
            priority_level = "low"
        
        mock_items.append({
            "id": i + 1,
            "candidate_id": i + 100,
            "doc_id": sent_data["doc_id"],
            "sent_id": f"s{i}",
            "text": text,
            "priority_score": priority_score,
            "priority_level": priority_level,
            "status": "pending",
            "entities": entities,
            "relations": [],
            "topics": [{"topic": "T_DISEASE", "score": 0.85}] if entities else [],
            "source": sent_data["source"]
        })
    
    # Save to file
    output_file = pipeline_root / "services/api/realistic_mock_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mock_items, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Saved {len(mock_items)} realistic items to: {output_file}")
    return True

def update_mock_api():
    """Update the mock API to use realistic data"""
    
    api_file = pipeline_root / "services/api/mock_api.py"
    
    # Add code to load realistic data at the top
    load_code = '''
# Load realistic mock data
try:
    with open("realistic_mock_data.json", "r", encoding="utf-8") as f:
        REALISTIC_MOCK_DATA = json.load(f)
    print(f"✅ Loaded {len(REALISTIC_MOCK_DATA)} realistic mock items")
except FileNotFoundError:
    REALISTIC_MOCK_DATA = []
    print("⚠️ No realistic mock data file found")
'''
    
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Insert after imports
    insert_pos = content.find('# Mock data generators')
    if insert_pos != -1:
        content = content[:insert_pos] + load_code + '\n' + content[insert_pos:]
    
    # Replace the triage queue function
    old_func_start = content.find('@app.get("/triage/queue")')
    old_func_end = content.find('\n@app.', old_func_start + 1)
    
    if old_func_start != -1 and old_func_end != -1:
        new_func = '''@app.get("/triage/queue")
async def get_triage_queue(
    status: Optional[str] = Query(None),
    sort_by: str = Query("priority"),
    limit: int = Query(50),
    offset: int = Query(0),
    search: Optional[str] = Query(None),
    priority_level: Optional[str] = Query(None)
):
    """Get triage queue items - now with realistic data from your documents"""
    
    # Use realistic data if available, otherwise fall back to generated
    if REALISTIC_MOCK_DATA:
        items = REALISTIC_MOCK_DATA.copy()
    else:
        # Fallback to generated data
        items = []
        for i in range(min(limit, 10)):
            priority_score = random.uniform(0.3, 0.98)
            priority_level_calc = "high" if priority_score > 0.8 else "medium" if priority_score > 0.5 else "low"
            
            items.append({
                "id": i + 1,
                "candidate_id": i + 100,
                "doc_id": f"test{(i % 3) + 1}",
                "sent_id": f"s{i}",
                "text": f"Real sentence {i+1} about shrimp aquaculture from imported documents.",
                "priority_score": priority_score,
                "priority_level": priority_level_calc,
                "status": "pending",
                "entities": generate_sample_entities("Penaeus vannamei Vibrio"),
                "relations": [],
                "topics": generate_sample_topics("Disease in shrimp"),
                "created_at": datetime.utcnow().isoformat()
            })
    
    # Apply filters
    if status:
        items = [item for item in items if item.get("status") == status]
    
    if priority_level:
        items = [item for item in items if item.get("priority_level") == priority_level]
    
    if search:
        items = [item for item in items if search.lower() in item.get("text", "").lower()]
    
    # Sort
    if sort_by == "priority":
        items.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
    else:
        items.sort(key=lambda x: x.get("id", 0))
    
    # Pagination
    total = len(items)
    items = items[offset:offset + limit]
    
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {
            "status": status,
            "search": search,
            "priority_level": priority_level,
            "sort_by": sort_by
        }
    }

'''
        
        content = content[:old_func_start] + new_func + content[old_func_end:]
        
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Mock API updated to use realistic data")
        return True
    
    return False

def main():
    """Create realistic mock data and update API"""
    
    print("🚀 Creating Realistic Mock Data from Your Documents\n")
    
    # Extract sentences
    if not create_mock_data_file():
        return 1
    
    # Update API
    if update_mock_api():
        print("\n✅ Realistic mock data created!")
        print("🔄 The API will auto-reload and show real sentences")
        print("💡 Refresh your browser to see actual content from your documents")
    else:
        print("\n❌ Failed to update API")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
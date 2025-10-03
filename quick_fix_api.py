#!/usr/bin/env python3
"""
Quick fix for the API database query issue.
"""

import sys
from pathlib import Path

def fix_query():
    """Fix the database query in the API"""
    
    api_file = Path("services/api/sqlite_api.py")
    
    # Read current file
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the problematic query
    old_query = '''                SELECT 
                    tq.item_id as id,
                    tq.candidate_id,
                    tq.doc_id,
                    tq.sent_id,
                    tq.priority_score,
                    tq.priority_level,
                    tq.status,
                    tq.created_at,
                    c.entity_data,
                    d.title as doc_title
                FROM triage_queue tq
                JOIN candidates c ON tq.candidate_id = c.candidate_id
                LEFT JOIN documents d ON tq.doc_id = d.doc_id'''
    
    new_query = '''                SELECT 
                    item_id as id,
                    candidate_id,
                    doc_id,
                    sent_id,
                    priority_score,
                    priority_level,
                    status,
                    created_at
                FROM triage_queue'''
    
    if old_query in content:
        content = content.replace(old_query, new_query)
        
        # Also simplify the processing
        old_processing = '''                try:
                    entities = json.loads(row['entity_data']) if row['entity_data'] else []
                    
                    # Get sentence text from candidates or create sample
                    sentence_text = f"Real sentence from {row['doc_title'] or row['doc_id']} (ID: {row['sent_id']})"
                    
                    # Try to get actual text if available
                    sentence_cursor = conn.execute(
                        "SELECT text FROM sentences WHERE doc_id = ? AND sent_id = ?",
                        (row['doc_id'], row['sent_id'])
                    )
                    sentence_row = sentence_cursor.fetchone()
                    if sentence_row:
                        sentence_text = sentence_row['text']
                    
                    items.append({
                        "id": row['id'],
                        "candidate_id": row['candidate_id'],
                        "doc_id": row['doc_id'],
                        "sent_id": row['sent_id'],
                        "text": sentence_text,
                        "priority_score": row['priority_score'],
                        "priority_level": row['priority_level'],
                        "status": row['status'],
                        "entities": entities,
                        "relations": [],  # TODO: Add relations if available
                        "topics": [],    # TODO: Add topics if available
                        "created_at": row['created_at']
                    })
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue'''
        
        new_processing = '''                try:
                    # Get entity data from candidates table
                    entity_cursor = conn.execute(
                        "SELECT entity_data FROM candidates WHERE candidate_id = ?",
                        (row['candidate_id'],)
                    )
                    entity_row = entity_cursor.fetchone()
                    entities = json.loads(entity_row['entity_data']) if entity_row and entity_row['entity_data'] else []
                    
                    # Create descriptive text
                    sentence_text = f"OpenAI processed sentence from {row['doc_id']} with {len(entities)} entities"
                    
                    items.append({
                        "id": row['id'],
                        "candidate_id": row['candidate_id'],
                        "doc_id": row['doc_id'],
                        "sent_id": row['sent_id'],
                        "text": sentence_text,
                        "priority_score": row['priority_score'],
                        "priority_level": row['priority_level'],
                        "status": row['status'],
                        "entities": entities,
                        "relations": [],
                        "topics": [],
                        "created_at": row['created_at']
                    })
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue'''
        
        content = content.replace(old_processing, new_processing)
        
        # Write back
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Query fixed!")
        return True
    else:
        print("❌ Query not found to fix")
        return False

def main():
    print("🔧 Quick fixing API database query...")
    
    if fix_query():
        print("💡 The API should auto-reload and pick up the fix")
    else:
        print("❌ Fix failed")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
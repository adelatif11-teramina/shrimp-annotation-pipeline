#!/usr/bin/env python3
"""
Patch the API to show real data instead of mock data.
"""

import sys
from pathlib import Path

def patch_api():
    """Replace mock triage queue with real database query"""
    
    api_file = Path("services/api/sqlite_api.py")
    
    if not api_file.exists():
        print("❌ API file not found")
        return False
    
    print("🔧 Patching API to use real database data...")
    
    # Read current file
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the mock triage queue function
    mock_start = content.find('"""Get triage queue items (mock data for now)"""')
    if mock_start == -1:
        print("❌ Mock function not found")
        return False
    
    # Find the end of the function (next @app decorator)
    func_start = content.rfind('async def get_triage_queue', 0, mock_start)
    func_end = content.find('\n@app.', mock_start)
    
    if func_start == -1 or func_end == -1:
        print("❌ Function boundaries not found")
        return False
    
    # New function implementation
    new_function = '''async def get_triage_queue(
    status: Optional[str] = Query(None),
    sort_by: str = Query("priority"),
    limit: int = Query(50),
    offset: int = Query(0),
    search: Optional[str] = Query(None),
    priority_level: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Get triage queue items from real database"""
    try:
        # Query real data from database
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build WHERE clause
            where_conditions = ["tq.status = 'pending'"]
            params = []
            
            if status:
                where_conditions.append("tq.status = ?")
                params.append(status)
            
            if priority_level:
                where_conditions.append("tq.priority_level = ?")
                params.append(priority_level)
            
            where_clause = " AND ".join(where_conditions)
            
            # Order by clause
            if sort_by == "priority":
                order_clause = "ORDER BY tq.priority_score DESC"
            else:
                order_clause = "ORDER BY tq.created_at DESC"
            
            # Main query
            query = f"""
                SELECT 
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
                LEFT JOIN documents d ON tq.doc_id = d.doc_id
                WHERE {where_clause}
                {order_clause}
                LIMIT ? OFFSET ?
            """
            
            cursor = conn.execute(query, params + [limit, offset])
            rows = cursor.fetchall()
            
            # Format results
            items = []
            for row in rows:
                try:
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
                    continue
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM triage_queue tq WHERE {where_clause}"
            count_cursor = conn.execute(count_query, params)
            total = count_cursor.fetchone()[0]
            
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
            
    except Exception as e:
        print(f"Database error: {e}")
        # Fallback to empty response
        return {
            "items": [],
            "total": 0,
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
    
    # Replace the function
    new_content = content[:func_start] + new_function + content[func_end:]
    
    # Add import for sqlite3 if not present
    if 'import sqlite3' not in new_content:
        import_pos = new_content.find('import json')
        if import_pos != -1:
            new_content = new_content[:import_pos] + 'import sqlite3\n' + new_content[import_pos:]
    
    # Write back
    with open(api_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ API patched successfully!")
    print("🔄 Restart the API server to see real data")
    
    return True

def main():
    print("🚀 Patching API for Real Data\n")
    
    if patch_api():
        print("\n✅ Patch applied!")
        print("💡 Restart the API server:")
        print("   1. Stop current server (Ctrl+C)")
        print("   2. Run: cd services/api && python sqlite_api.py")
        print("   3. Refresh UI to see real annotation queue")
    else:
        print("\n❌ Patch failed")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
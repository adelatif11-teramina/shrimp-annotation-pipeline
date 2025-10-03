"""
Simple SQLite Database Implementation
No external dependencies - uses only Python's built-in sqlite3
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

class SimpleDatabase:
    def __init__(self, db_path: str = "annotations.db"):
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    email TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    title TEXT,
                    source TEXT NOT NULL,
                    raw_text TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sentences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sent_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    processed BOOLEAN DEFAULT 0,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
            """)
            
            # Candidates table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    sent_id TEXT NOT NULL,
                    sentence_id INTEGER,
                    source TEXT NOT NULL,
                    candidate_type TEXT NOT NULL,
                    entities TEXT,
                    relations TEXT,
                    topics TEXT,
                    confidence REAL,
                    priority_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT 0,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                    FOREIGN KEY (sentence_id) REFERENCES sentences(id)
                )
            """)
            
            # Triage queue table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS triage_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    priority_score REAL NOT NULL,
                    priority_level TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    assigned_to INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    assigned_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
                    FOREIGN KEY (assigned_to) REFERENCES users(id)
                )
            """)
            
            # Annotations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    sent_id TEXT NOT NULL,
                    candidate_id INTEGER,
                    user_id INTEGER NOT NULL,
                    annotation_type TEXT NOT NULL,
                    entities TEXT,
                    relations TEXT,
                    topics TEXT,
                    decision TEXT NOT NULL,
                    confidence REAL,
                    notes TEXT,
                    time_spent REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Annotation history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    annotation_id INTEGER NOT NULL,
                    candidate_id INTEGER,
                    version INTEGER NOT NULL,
                    previous_state TEXT,
                    new_state TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    changed_by INTEGER NOT NULL,
                    changed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    change_reason TEXT,
                    FOREIGN KEY (annotation_id) REFERENCES annotations(id),
                    FOREIGN KEY (changed_by) REFERENCES users(id)
                )
            """)
            
            # System stats table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_documents INTEGER DEFAULT 0,
                    total_sentences INTEGER DEFAULT 0,
                    processed_sentences INTEGER DEFAULT 0,
                    total_candidates INTEGER DEFAULT 0,
                    total_annotations INTEGER DEFAULT 0,
                    accepted_annotations INTEGER DEFAULT 0,
                    rejected_annotations INTEGER DEFAULT 0,
                    queue_size INTEGER DEFAULT 0,
                    average_priority REAL DEFAULT 0.0,
                    annotations_per_hour REAL DEFAULT 0.0,
                    average_confidence REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def create_default_users(self):
        """Create default users if they don't exist"""
        default_users = [
            {"username": "admin", "token": "local-admin-2024", "role": "admin", "email": "admin@local.dev"},
            {"username": "annotator1", "token": "anno-team-001", "role": "annotator", "email": "ann1@local.dev"},
            {"username": "annotator2", "token": "anno-team-002", "role": "annotator", "email": "ann2@local.dev"},
            {"username": "reviewer", "token": "review-lead-003", "role": "reviewer", "email": "review@local.dev"},
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for user_data in default_users:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO users (username, token, role, email)
                        VALUES (?, ?, ?, ?)
                    """, (user_data["username"], user_data["token"], user_data["role"], user_data["email"]))
                except sqlite3.Error as e:
                    print(f"Error creating user {user_data['username']}: {e}")
            conn.commit()
    
    def get_user_by_token(self, token: str) -> Optional[Dict]:
        """Get user by authentication token"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, username, role, email FROM users 
                WHERE token = ? AND is_active = 1
            """, (token,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def create_document(self, doc_data: Dict) -> int:
        """Create a new document"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO documents (doc_id, title, source, raw_text, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                doc_data["doc_id"],
                doc_data.get("title"),
                doc_data["source"],
                doc_data.get("raw_text"),
                json.dumps(doc_data.get("metadata", {}))
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_documents(self, limit: int = 50, offset: int = 0, search: str = None, source: str = None) -> Dict:
        """Get documents with filtering"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build query
            where_conditions = []
            params = []
            
            if search:
                where_conditions.append("(title LIKE ? OR doc_id LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])
            
            if source:
                where_conditions.append("source = ?")
                params.append(source)
            
            where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Get total count
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM documents{where_clause}", params)
            total = count_cursor.fetchone()[0]
            
            # Get documents
            params.extend([limit, offset])
            cursor = conn.execute(f"""
                SELECT id, doc_id, title, source, created_at
                FROM documents{where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params)
            
            documents = [dict(row) for row in cursor.fetchall()]
            
            return {
                "documents": documents,
                "total": total,
                "limit": limit,
                "offset": offset
            }
    
    def create_annotation(self, annotation_data: Dict) -> int:
        """Create annotation with history tracking"""
        with sqlite3.connect(self.db_path) as conn:
            timestamp = datetime.utcnow().isoformat()
            
            # Check for existing annotation
            cursor = conn.execute("""
                SELECT id, decision, entities, relations, topics, confidence, notes, version
                FROM annotations WHERE candidate_id = ?
            """, (annotation_data["candidate_id"],))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing annotation
                annotation_id = existing[0]
                new_version = existing[7] + 1
                
                # Create history entry
                conn.execute("""
                    INSERT INTO annotation_history 
                    (annotation_id, candidate_id, version, previous_state, new_state, 
                     change_type, changed_by, change_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    annotation_id,
                    annotation_data["candidate_id"],
                    new_version,
                    json.dumps({
                        "decision": existing[1],
                        "entities": json.loads(existing[2] or "[]"),
                        "relations": json.loads(existing[3] or "[]"),
                        "topics": json.loads(existing[4] or "[]"),
                        "confidence": existing[5],
                        "notes": existing[6]
                    }),
                    json.dumps({
                        "decision": annotation_data["decision"],
                        "entities": annotation_data.get("entities", []),
                        "relations": annotation_data.get("relations", []),
                        "topics": annotation_data.get("topics", []),
                        "confidence": annotation_data.get("confidence", 0.8),
                        "notes": annotation_data.get("notes", "")
                    }),
                    "modify",
                    annotation_data["user_id"],
                    f"Updated from {existing[1]} to {annotation_data['decision']}"
                ))
                
                # Update annotation
                conn.execute("""
                    UPDATE annotations SET
                    decision = ?, entities = ?, relations = ?, topics = ?,
                    confidence = ?, notes = ?, updated_at = ?, version = ?
                    WHERE id = ?
                """, (
                    annotation_data["decision"],
                    json.dumps(annotation_data.get("entities", [])),
                    json.dumps(annotation_data.get("relations", [])),
                    json.dumps(annotation_data.get("topics", [])),
                    annotation_data.get("confidence", 0.8),
                    annotation_data.get("notes", ""),
                    timestamp,
                    new_version,
                    annotation_id
                ))
                
            else:
                # Create new annotation
                cursor = conn.execute("""
                    INSERT INTO annotations 
                    (doc_id, sent_id, candidate_id, user_id, annotation_type,
                     entities, relations, topics, decision, confidence, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    annotation_data.get("doc_id", ""),
                    annotation_data.get("sent_id", ""),
                    annotation_data["candidate_id"],
                    annotation_data["user_id"],
                    annotation_data.get("annotation_type", "combined"),
                    json.dumps(annotation_data.get("entities", [])),
                    json.dumps(annotation_data.get("relations", [])),
                    json.dumps(annotation_data.get("topics", [])),
                    annotation_data["decision"],
                    annotation_data.get("confidence", 0.8),
                    annotation_data.get("notes", "")
                ))
                annotation_id = cursor.lastrowid
                
                # Create initial history entry
                conn.execute("""
                    INSERT INTO annotation_history 
                    (annotation_id, candidate_id, version, new_state, 
                     change_type, changed_by, change_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    annotation_id,
                    annotation_data["candidate_id"],
                    1,
                    json.dumps({
                        "decision": annotation_data["decision"],
                        "entities": annotation_data.get("entities", []),
                        "relations": annotation_data.get("relations", []),
                        "topics": annotation_data.get("topics", []),
                        "confidence": annotation_data.get("confidence", 0.8),
                        "notes": annotation_data.get("notes", "")
                    }),
                    "create",
                    annotation_data["user_id"],
                    "Initial annotation"
                ))
            
            conn.commit()
            return annotation_id
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Document stats
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            stats["total_documents"] = cursor.fetchone()[0]
            
            # Sentence stats
            cursor = conn.execute("SELECT COUNT(*) FROM sentences")
            stats["total_sentences"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM sentences WHERE processed = 1")
            stats["processed_sentences"] = cursor.fetchone()[0]
            
            # Annotation stats
            cursor = conn.execute("SELECT COUNT(*) FROM annotations")
            stats["total_annotations"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM annotations WHERE decision = 'accept'")
            stats["accepted_annotations"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM annotations WHERE decision = 'reject'")
            stats["rejected_annotations"] = cursor.fetchone()[0]
            
            # Queue stats
            cursor = conn.execute("SELECT COUNT(*) FROM triage_queue WHERE status = 'pending'")
            stats["queue_size"] = cursor.fetchone()[0]
            
            # Calculate averages
            cursor = conn.execute("SELECT AVG(confidence) FROM annotations")
            avg_conf = cursor.fetchone()[0]
            stats["average_confidence"] = round(avg_conf or 0.0, 3)
            
            cursor = conn.execute("SELECT AVG(priority_score) FROM triage_queue")
            avg_pri = cursor.fetchone()[0]
            stats["average_priority"] = round(avg_pri or 0.0, 3)
            
            return stats
    
    def close(self):
        """Close database connection"""
        pass  # sqlite3 auto-closes connections

# Global database instance
db = SimpleDatabase()
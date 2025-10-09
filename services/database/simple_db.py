"""Simple SQLite database helper used in tests and local tooling."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimpleDatabase:
    """Lightweight SQLite wrapper with helpers for tests."""

    def __init__(self, db_path: str = "data/local/annotations.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    email TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
                """
            )

            conn.execute(
                """
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
                """
            )

            conn.execute(
                """
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
                """
            )

            conn.execute(
                """
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
                    processed BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                    FOREIGN KEY (sentence_id) REFERENCES sentences(id)
                )
                """
            )

            conn.execute(
                """
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
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

            conn.execute(
                """
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
                """
            )

            conn.execute(
                """
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
                """
            )

            conn.commit()

    def create_default_users(self) -> None:
        """Populate a handful of default users for local development."""
        defaults = [
            {"username": "admin", "token": "local-admin-2024", "role": "admin", "email": "admin@local.dev"},
            {"username": "annotator1", "token": "anno-team-001", "role": "annotator", "email": "ann1@local.dev"},
            {"username": "annotator2", "token": "anno-team-002", "role": "annotator", "email": "ann2@local.dev"},
            {"username": "reviewer", "token": "review-lead-003", "role": "reviewer", "email": "review@local.dev"},
        ]

        with sqlite3.connect(self.db_path) as conn:
            for user in defaults:
                conn.execute(
                    "INSERT OR IGNORE INTO users (username, token, role, email) VALUES (?, ?, ?, ?)",
                    (user["username"], user["token"], user["role"], user["email"]),
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json(value: Optional[Dict[str, Any]]) -> str:
        return json.dumps(value or {})

    @staticmethod
    def _json_list(value: Optional[List[Any]]) -> str:
        return json.dumps(value or [])

    @staticmethod
    def _parse_json(value: Optional[str], default: Any) -> Any:
        if value in (None, "", b""):
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    def _fetch_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, username, token, role, email, is_active, created_at FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def create_user(
        self,
        *,
        username: str,
        token: str,
        role: str,
        email: Optional[str] = None,
        is_active: bool = True,
    ) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, token, role, email, is_active) VALUES (?, ?, ?, ?, ?)",
                (username, token, role, email, int(is_active))
            )
            conn.commit()
            return self._fetch_user(cursor.lastrowid)

    def update_user(self, user_id: int, **updates) -> Optional[Dict[str, Any]]:
        allowed = {"username", "token", "role", "email", "is_active"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return self._fetch_user(user_id)

        set_clause = ", ".join(f"{field} = ?" for field in fields)
        params = [int(v) if field == "is_active" else v for field, v in fields.items()]
        params.append(user_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", params)
            conn.commit()
        return self._fetch_user(user_id)

    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, username, role, email FROM users WHERE token = ? AND is_active = 1",
                (token,)
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Document & sentence operations
    # ------------------------------------------------------------------

    def create_document(
        self,
        *,
        doc_id: str,
        title: Optional[str] = None,
        source: str,
        raw_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, title, source, raw_text, metadata) VALUES (?, ?, ?, ?, ?)",
                (doc_id, title, source, raw_text, self._json(metadata))
            )
            conn.commit()
        return self.get_document(doc_id)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, doc_id, title, source, raw_text, metadata, created_at, updated_at FROM documents WHERE doc_id = ?",
                (doc_id,)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            data["metadata"] = self._parse_json(data.get("metadata"), {})
            return data

    def delete_document(self, doc_id: str) -> Dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
        return {"status": "deleted", "doc_id": doc_id}

    def get_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            where_clauses: List[str] = []
            params: List[Any] = []

            if search:
                where_clauses.append("(title LIKE ? OR doc_id LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])

            if source:
                where_clauses.append("source = ?")
                params.append(source)

            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            total = conn.execute(f"SELECT COUNT(*) FROM documents{where_sql}", params).fetchone()[0]

            params_with_paging = params + [limit, offset]
            rows = conn.execute(
                f"""
                SELECT id, doc_id, title, source, created_at
                FROM documents{where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params_with_paging,
            ).fetchall()

            return {
                "documents": [dict(row) for row in rows],
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    def create_sentence(
        self,
        *,
        sent_id: str,
        doc_id: str,
        text: str,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        processed: bool = False,
    ) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO sentences (sent_id, doc_id, text, start_offset, end_offset, processed)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sent_id, doc_id, text, start_offset, end_offset, int(processed))
            )
            conn.commit()
            row_id = cursor.lastrowid
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, sent_id, doc_id, text, start_offset, end_offset, processed FROM sentences WHERE id = ?",
                (row_id,)
            ).fetchone()
        data = dict(row)
        data["processed"] = bool(data["processed"])
        return data

    def get_sentences_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, sent_id, doc_id, text, start_offset, end_offset, processed FROM sentences WHERE doc_id = ? ORDER BY id",
                (doc_id,)
            ).fetchall()
        sentences: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["processed"] = bool(data["processed"])
            sentences.append(data)
        return sentences

    # ------------------------------------------------------------------
    # Candidate operations
    # ------------------------------------------------------------------

    def _fetch_candidate(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, doc_id, sent_id, sentence_id, source, candidate_type, entities,
                       relations, topics, confidence, priority_score, processed, created_at
                FROM candidates WHERE id = ?
                """,
                (candidate_id,)
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["entities"] = self._parse_json(data.get("entities"), [])
        data["relations"] = self._parse_json(data.get("relations"), [])
        data["topics"] = self._parse_json(data.get("topics"), [])
        data["processed"] = bool(data.get("processed", 0))
        return data

    def create_candidate(self, **candidate_data) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO candidates (doc_id, sent_id, sentence_id, source, candidate_type,
                                        entities, relations, topics, confidence, priority_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_data["doc_id"],
                    candidate_data["sent_id"],
                    candidate_data.get("sentence_id"),
                    candidate_data.get("source", "unknown"),
                    candidate_data.get("candidate_type", "entity"),
                    self._json_list(candidate_data.get("entities")),
                    self._json_list(candidate_data.get("relations")),
                    self._json_list(candidate_data.get("topics")),
                    candidate_data.get("confidence"),
                    candidate_data.get("priority_score"),
                ),
            )
            conn.commit()
            candidate_id = cursor.lastrowid
        return self._fetch_candidate(candidate_id)

    # ------------------------------------------------------------------
    # Annotation operations
    # ------------------------------------------------------------------

    def _fetch_annotation(self, annotation_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, doc_id, sent_id, candidate_id, user_id, annotation_type,
                       entities, relations, topics, decision, confidence, notes,
                       time_spent, version, created_at, updated_at
                FROM annotations WHERE id = ?
                """,
                (annotation_id,)
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["entities"] = self._parse_json(data.get("entities"), [])
        data["relations"] = self._parse_json(data.get("relations"), [])
        data["topics"] = self._parse_json(data.get("topics"), [])
        return data

    def create_annotation(self, **annotation_data: Any) -> Dict[str, Any]:
        annotation_data = dict(annotation_data)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            timestamp = datetime.utcnow().isoformat()

            existing = conn.execute(
                """
                SELECT id, decision, entities, relations, topics, confidence, notes, version
                FROM annotations WHERE candidate_id = ?
                """,
                (annotation_data["candidate_id"],),
            ).fetchone()

            if existing:
                annotation_id = existing[0]
                new_version = existing[7] + 1

                conn.execute(
                    """
                    INSERT INTO annotation_history (
                        annotation_id, candidate_id, version, previous_state, new_state,
                        change_type, changed_by, change_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        annotation_id,
                        annotation_data["candidate_id"],
                        new_version,
                        json.dumps(
                            {
                                "decision": existing[1],
                                "entities": self._parse_json(existing[2], []),
                                "relations": self._parse_json(existing[3], []),
                                "topics": self._parse_json(existing[4], []),
                                "confidence": existing[5],
                                "notes": existing[6],
                            }
                        ),
                        json.dumps(
                            {
                                "decision": annotation_data["decision"],
                                "entities": annotation_data.get("entities", []),
                                "relations": annotation_data.get("relations", []),
                                "topics": annotation_data.get("topics", []),
                                "confidence": annotation_data.get("confidence", 0.8),
                                "notes": annotation_data.get("notes", ""),
                            }
                        ),
                        "modify",
                        annotation_data["user_id"],
                        f"Updated from {existing[1]} to {annotation_data['decision']}",
                    ),
                )

                conn.execute(
                    """
                    UPDATE annotations
                    SET decision = ?, entities = ?, relations = ?, topics = ?,
                        confidence = ?, notes = ?, updated_at = ?, version = ?
                    WHERE id = ?
                    """,
                    (
                        annotation_data["decision"],
                        self._json_list(annotation_data.get("entities")),
                        self._json_list(annotation_data.get("relations")),
                        self._json_list(annotation_data.get("topics")),
                        annotation_data.get("confidence", 0.8),
                        annotation_data.get("notes", ""),
                        timestamp,
                        new_version,
                        annotation_id,
                    ),
                )

            else:
                cursor = conn.execute(
                    """
                    INSERT INTO annotations (
                        doc_id, sent_id, candidate_id, user_id, annotation_type,
                        entities, relations, topics, decision, confidence, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        annotation_data.get("doc_id", ""),
                        annotation_data.get("sent_id", ""),
                        annotation_data["candidate_id"],
                        annotation_data["user_id"],
                        annotation_data.get("annotation_type", "combined"),
                        self._json_list(annotation_data.get("entities")),
                        self._json_list(annotation_data.get("relations")),
                        self._json_list(annotation_data.get("topics")),
                        annotation_data["decision"],
                        annotation_data.get("confidence", 0.8),
                        annotation_data.get("notes", ""),
                    ),
                )
                annotation_id = cursor.lastrowid

                conn.execute(
                    """
                    INSERT INTO annotation_history (
                        annotation_id, candidate_id, version, new_state, change_type, changed_by, change_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        annotation_id,
                        annotation_data["candidate_id"],
                        1,
                        json.dumps(
                            {
                                "decision": annotation_data["decision"],
                                "entities": annotation_data.get("entities", []),
                                "relations": annotation_data.get("relations", []),
                                "topics": annotation_data.get("topics", []),
                                "confidence": annotation_data.get("confidence", 0.8),
                                "notes": annotation_data.get("notes", ""),
                            }
                        ),
                        "create",
                        annotation_data["user_id"],
                        "Initial annotation",
                    ),
                )

            conn.commit()
        return self._fetch_annotation(annotation_id)

    def get_annotations(self, *, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM annotations"
        params: List[Any] = []
        if doc_id:
            query += " WHERE doc_id = ?"
            params.append(doc_id)
        query += " ORDER BY created_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        annotations = []
        for row in rows:
            data = dict(row)
            data["entities"] = self._parse_json(data.get("entities"), [])
            data["relations"] = self._parse_json(data.get("relations"), [])
            data["topics"] = self._parse_json(data.get("topics"), [])
            annotations.append(data)
        return annotations

    def get_user_annotations(self, user_id: int) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM annotations WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            ).fetchall()
        annotations = []
        for row in rows:
            data = dict(row)
            data["entities"] = self._parse_json(data.get("entities"), [])
            data["relations"] = self._parse_json(data.get("relations"), [])
            data["topics"] = self._parse_json(data.get("topics"), [])
            annotations.append(data)
        return annotations

    # ------------------------------------------------------------------
    # Triage operations
    # ------------------------------------------------------------------

    def add_to_triage_queue(
        self,
        *,
        candidate_id: int,
        priority_score: float,
        priority_level: str,
        status: str = "pending",
        assigned_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO triage_queue (candidate_id, priority_score, priority_level, status, assigned_to)
                VALUES (?, ?, ?, ?, ?)
                """,
                (candidate_id, priority_score, priority_level, status, assigned_to)
            )
            conn.commit()
            item_id = cursor.lastrowid
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM triage_queue WHERE id = ?", (item_id,)).fetchone()
        return dict(row)

    def get_triage_queue(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM triage_queue ORDER BY priority_score DESC, created_at ASC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def update_triage_status(self, item_id: int, status: str) -> None:
        timestamp_column = "completed_at" if status == "completed" else "assigned_at"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE triage_queue SET status = ?, {timestamp_column} = CURRENT_TIMESTAMP WHERE id = ?",
                (status, item_id)
            )
            conn.commit()

    def get_triage_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM triage_queue WHERE id = ?", (item_id,)).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            stats: Dict[str, Any] = {}

            stats["total_documents"] = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            stats["total_sentences"] = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
            stats["processed_sentences"] = conn.execute(
                "SELECT COUNT(*) FROM sentences WHERE processed = 1"
            ).fetchone()[0]
            stats["total_annotations"] = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
            stats["accepted_annotations"] = conn.execute(
                "SELECT COUNT(*) FROM annotations WHERE decision = 'accepted'"
            ).fetchone()[0]
            stats["rejected_annotations"] = conn.execute(
                "SELECT COUNT(*) FROM annotations WHERE decision = 'rejected'"
            ).fetchone()[0]
            stats["queue_size"] = conn.execute(
                "SELECT COUNT(*) FROM triage_queue WHERE status = 'pending'"
            ).fetchone()[0]
            stats["total_users"] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

            avg_conf = conn.execute("SELECT AVG(confidence) FROM annotations").fetchone()[0]
            stats["average_confidence"] = round(avg_conf or 0.0, 3)

            avg_priority = conn.execute("SELECT AVG(priority_score) FROM triage_queue").fetchone()[0]
            stats["average_priority"] = round(avg_priority or 0.0, 3)

            return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:  # pragma: no cover - sqlite connections auto-close
        pass


# Default instance used by legacy scripts
db = SimpleDatabase()

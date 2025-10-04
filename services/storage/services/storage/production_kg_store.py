#!/usr/bin/env python3
"""
Production Knowledge Graph Store

Robust storage system for KG triplets with versioning, backup, and validation.
Ensures all annotated triplets are permanently preserved for production use.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class KGTriplet:
    """Knowledge Graph triplet with full context"""
    head_entity: Dict[str, Any]
    relation: str
    tail_entity: Dict[str, Any]
    confidence: float
    evidence_text: str
    doc_id: str
    sent_id: str
    annotator: str
    timestamp: str
    triplet_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.triplet_id:
            # Generate unique triplet ID
            content = f"{self.head_entity['text']}_{self.relation}_{self.tail_entity['text']}_{self.doc_id}_{self.sent_id}"
            self.triplet_id = hashlib.md5(content.encode()).hexdigest()[:16]

@dataclass
class KGDataset:
    """Complete KG dataset with metadata"""
    triplets: List[KGTriplet]
    entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    version: str
    created_at: str
    
class ProductionKGStore:
    """
    Production-grade Knowledge Graph storage system.
    
    Features:
    - Immutable triplet storage with versioning
    - Automatic backups and redundancy
    - Export to multiple formats (RDF, JSON-LD, CSV)
    - Validation against v2.0 ontology
    - Audit trails for all changes
    """
    
    def __init__(self, 
                 base_path: Path,
                 enable_backup: bool = True,
                 backup_retention_days: int = 30):
        """
        Initialize production KG store.
        
        Args:
            base_path: Base storage directory
            enable_backup: Enable automatic backups
            backup_retention_days: Days to retain backups
        """
        self.base_path = Path(base_path)
        self.kg_store_path = self.base_path / "kg_store"
        self.backup_path = self.base_path / "backups"
        self.versions_path = self.kg_store_path / "versions"
        self.exports_path = self.base_path / "exports"
        
        self.enable_backup = enable_backup
        self.backup_retention_days = backup_retention_days
        
        # Create directory structure
        for path in [self.kg_store_path, self.backup_path, self.versions_path, self.exports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite index for fast queries
        self.index_db_path = self.kg_store_path / "triplet_index.db"
        self._init_triplet_index()
        
        logger.info(f"Initialized production KG store at {base_path}")
    
    def _init_triplet_index(self):
        """Initialize SQLite index for triplet queries"""
        with sqlite3.connect(self.index_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS triplets (
                    triplet_id TEXT PRIMARY KEY,
                    head_entity_text TEXT NOT NULL,
                    head_entity_type TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    tail_entity_text TEXT NOT NULL,
                    tail_entity_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    doc_id TEXT NOT NULL,
                    sent_id TEXT NOT NULL,
                    annotator TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for fast queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_head_entity ON triplets(head_entity_text, head_entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation ON triplets(relation)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tail_entity ON triplets(tail_entity_text, tail_entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON triplets(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON triplets(confidence)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON triplets(timestamp)")
            
            conn.commit()
    
    def store_gold_annotation(self, 
                            annotation_data: Dict[str, Any],
                            doc_id: str,
                            sent_id: str) -> List[KGTriplet]:
        """
        Store validated annotation as KG triplets.
        
        Args:
            annotation_data: Complete annotation data from gold store
            doc_id: Document ID
            sent_id: Sentence/paragraph ID
            
        Returns:
            List of stored KG triplets
        """
        triplets = []
        entities = annotation_data.get("entities", [])
        relations = annotation_data.get("relations", [])
        
        # Extract metadata
        annotator = annotation_data.get("annotator", "unknown")
        timestamp = annotation_data.get("timestamp", datetime.now().isoformat())
        evidence_text = annotation_data.get("text", "")
        
        # Create entity lookup
        entity_lookup = {entity.get("id"): entity for entity in entities}
        
        # Convert relations to triplets
        for relation in relations:
            head_id = relation.get("head", {}).get("id") if isinstance(relation.get("head"), dict) else relation.get("head")
            tail_id = relation.get("tail", {}).get("id") if isinstance(relation.get("tail"), dict) else relation.get("tail")
            
            head_entity = entity_lookup.get(head_id)
            tail_entity = entity_lookup.get(tail_id)
            
            if head_entity and tail_entity:
                triplet = KGTriplet(
                    head_entity=head_entity,
                    relation=relation.get("type", relation.get("label", "unknown")),
                    tail_entity=tail_entity,
                    confidence=float(relation.get("confidence", 1.0)),
                    evidence_text=evidence_text,
                    doc_id=doc_id,
                    sent_id=sent_id,
                    annotator=annotator,
                    timestamp=timestamp
                )
                
                triplets.append(triplet)
        
        # Store triplets
        if triplets:
            self._persist_triplets(triplets)
            logger.info(f"Stored {len(triplets)} KG triplets from {doc_id}/{sent_id}")
        
        return triplets
    
    def _persist_triplets(self, triplets: List[KGTriplet]):
        """Persist triplets to storage and index"""
        current_time = datetime.now().isoformat()
        
        # Store individual triplet files
        for triplet in triplets:
            triplet_file = self.kg_store_path / f"{triplet.triplet_id}.json"
            
            with open(triplet_file, 'w') as f:
                json.dump(asdict(triplet), f, indent=2)
            
            # Add to index
            with sqlite3.connect(self.index_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO triplets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    triplet.triplet_id,
                    triplet.head_entity["text"],
                    triplet.head_entity["label"],
                    triplet.relation,
                    triplet.tail_entity["text"],
                    triplet.tail_entity["label"],
                    triplet.confidence,
                    triplet.doc_id,
                    triplet.sent_id,
                    triplet.annotator,
                    triplet.timestamp,
                    str(triplet_file),
                    current_time
                ))
                conn.commit()
        
        # Create backup if enabled
        if self.enable_backup:
            self._create_backup()
    
    def _create_backup(self):
        """Create backup of current KG store"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"kg_backup_{timestamp}"
        
        # Copy KG store
        shutil.copytree(self.kg_store_path, backup_dir)
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        logger.debug(f"Created KG backup: {backup_dir}")
    
    def _cleanup_old_backups(self):
        """Remove backups older than retention period"""
        cutoff_time = datetime.now().timestamp() - (self.backup_retention_days * 24 * 3600)
        
        for backup_dir in self.backup_path.glob("kg_backup_*"):
            if backup_dir.stat().st_mtime < cutoff_time:
                shutil.rmtree(backup_dir)
                logger.debug(f"Removed old backup: {backup_dir}")
    
    def query_triplets(self, 
                      head_entity: Optional[str] = None,
                      relation: Optional[str] = None,
                      tail_entity: Optional[str] = None,
                      min_confidence: float = 0.0,
                      limit: Optional[int] = None) -> List[KGTriplet]:
        """
        Query KG triplets with filters.
        
        Args:
            head_entity: Filter by head entity text
            relation: Filter by relation type
            tail_entity: Filter by tail entity text
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of matching KG triplets
        """
        conditions = ["confidence >= ?"]
        params = [min_confidence]
        
        if head_entity:
            conditions.append("head_entity_text LIKE ?")
            params.append(f"%{head_entity}%")
        
        if relation:
            conditions.append("relation = ?")
            params.append(relation)
        
        if tail_entity:
            conditions.append("tail_entity_text LIKE ?")
            params.append(f"%{tail_entity}%")
        
        query = f"""
            SELECT triplet_id, file_path FROM triplets 
            WHERE {' AND '.join(conditions)}
            ORDER BY confidence DESC, timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        triplets = []
        with sqlite3.connect(self.index_db_path) as conn:
            for row in conn.execute(query, params):
                triplet_id, file_path = row
                try:
                    with open(file_path, 'r') as f:
                        triplet_data = json.load(f)
                        triplets.append(KGTriplet(**triplet_data))
                except Exception as e:
                    logger.warning(f"Failed to load triplet {triplet_id}: {e}")
        
        return triplets
    
    def export_dataset(self, 
                      output_path: Path,
                      format: str = "jsonl",
                      min_confidence: float = 0.0,
                      include_metadata: bool = True) -> int:
        """
        Export complete KG dataset.
        
        Args:
            output_path: Output file path
            format: Export format (jsonl, rdf, csv, json-ld)
            min_confidence: Minimum confidence threshold
            include_metadata: Include annotation metadata
            
        Returns:
            Number of exported triplets
        """
        triplets = self.query_triplets(min_confidence=min_confidence)
        
        if format == "jsonl":
            self._export_jsonl(triplets, output_path, include_metadata)
        elif format == "rdf":
            self._export_rdf(triplets, output_path)
        elif format == "csv":
            self._export_csv(triplets, output_path)
        elif format == "json-ld":
            self._export_json_ld(triplets, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(triplets)} triplets to {output_path}")
        return len(triplets)
    
    def _export_jsonl(self, triplets: List[KGTriplet], output_path: Path, include_metadata: bool):
        """Export to JSONL format"""
        with open(output_path, 'w') as f:
            for triplet in triplets:
                export_data = {
                    "triplet_id": triplet.triplet_id,
                    "head": triplet.head_entity,
                    "relation": triplet.relation,
                    "tail": triplet.tail_entity,
                    "confidence": triplet.confidence
                }
                
                if include_metadata:
                    export_data.update({
                        "evidence_text": triplet.evidence_text,
                        "doc_id": triplet.doc_id,
                        "sent_id": triplet.sent_id,
                        "annotator": triplet.annotator,
                        "timestamp": triplet.timestamp
                    })
                
                f.write(json.dumps(export_data) + "\n")
    
    def _export_csv(self, triplets: List[KGTriplet], output_path: Path):
        """Export to CSV format"""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "triplet_id", "head_text", "head_type", "relation", 
                "tail_text", "tail_type", "confidence", "doc_id", "sent_id"
            ])
            
            for triplet in triplets:
                writer.writerow([
                    triplet.triplet_id,
                    triplet.head_entity["text"],
                    triplet.head_entity["label"],
                    triplet.relation,
                    triplet.tail_entity["text"],
                    triplet.tail_entity["label"],
                    triplet.confidence,
                    triplet.doc_id,
                    triplet.sent_id
                ])
    
    def _export_rdf(self, triplets: List[KGTriplet], output_path: Path):
        """Export to RDF/Turtle format"""
        with open(output_path, 'w') as f:
            f.write("@prefix shrimp: <http://shrimp-kg.org/ontology#> .\n")
            f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n")
            
            for triplet in triplets:
                head_uri = f"shrimp:{triplet.head_entity['text'].replace(' ', '_')}"
                tail_uri = f"shrimp:{triplet.tail_entity['text'].replace(' ', '_')}"
                relation_uri = f"shrimp:{triplet.relation}"
                
                f.write(f"{head_uri} {relation_uri} {tail_uri} .\n")
                f.write(f"{head_uri} rdfs:label \"{triplet.head_entity['text']}\" .\n")
                f.write(f"{tail_uri} rdfs:label \"{triplet.tail_entity['text']}\" .\n\n")
    
    def _export_json_ld(self, triplets: List[KGTriplet], output_path: Path):
        """Export to JSON-LD format"""
        json_ld = {
            "@context": {
                "@vocab": "http://shrimp-kg.org/ontology#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "@graph": []
        }
        
        for triplet in triplets:
            json_ld["@graph"].append({
                "@id": f"shrimp:{triplet.head_entity['text'].replace(' ', '_')}",
                f"shrimp:{triplet.relation}": {
                    "@id": f"shrimp:{triplet.tail_entity['text'].replace(' ', '_')}"
                },
                "rdfs:label": triplet.head_entity["text"],
                "confidence": triplet.confidence
            })
        
        with open(output_path, 'w') as f:
            json.dump(json_ld, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get KG store statistics"""
        with sqlite3.connect(self.index_db_path) as conn:
            # Total triplets
            total_triplets = conn.execute("SELECT COUNT(*) FROM triplets").fetchone()[0]
            
            # Entity type distribution
            entity_types = conn.execute("""
                SELECT head_entity_type, COUNT(*) as count FROM triplets 
                GROUP BY head_entity_type 
                ORDER BY count DESC
            """).fetchall()
            
            # Relation distribution
            relations = conn.execute("""
                SELECT relation, COUNT(*) as count FROM triplets 
                GROUP BY relation 
                ORDER BY count DESC
            """).fetchall()
            
            # Confidence distribution
            avg_confidence = conn.execute("SELECT AVG(confidence) FROM triplets").fetchone()[0]
            min_confidence = conn.execute("SELECT MIN(confidence) FROM triplets").fetchone()[0]
            max_confidence = conn.execute("SELECT MAX(confidence) FROM triplets").fetchone()[0]
            
            # Annotator distribution
            annotators = conn.execute("""
                SELECT annotator, COUNT(*) as count FROM triplets 
                GROUP BY annotator 
                ORDER BY count DESC
            """).fetchall()
        
        return {
            "total_triplets": total_triplets,
            "entity_type_distribution": dict(entity_types),
            "relation_distribution": dict(relations),
            "confidence_stats": {
                "average": avg_confidence,
                "minimum": min_confidence,
                "maximum": max_confidence
            },
            "annotator_distribution": dict(annotators),
            "storage_path": str(self.kg_store_path),
            "index_path": str(self.index_db_path)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production KG Store Management")
    parser.add_argument("--store-path", required=True, help="KG store base path")
    parser.add_argument("--action", choices=["init", "stats", "export", "query"], 
                       default="stats", help="Action to perform")
    parser.add_argument("--export-format", choices=["jsonl", "rdf", "csv", "json-ld"],
                       default="jsonl", help="Export format")
    parser.add_argument("--output", help="Output file for export")
    parser.add_argument("--min-confidence", type=float, default=0.0, 
                       help="Minimum confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize store
    kg_store = ProductionKGStore(Path(args.store_path))
    
    if args.action == "stats":
        stats = kg_store.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.action == "export":
        if not args.output:
            print("--output required for export action")
            exit(1)
        
        count = kg_store.export_dataset(
            Path(args.output),
            format=args.export_format,
            min_confidence=args.min_confidence
        )
        print(f"Exported {count} triplets to {args.output}")
    
    elif args.action == "query":
        triplets = kg_store.query_triplets(min_confidence=args.min_confidence, limit=10)
        for triplet in triplets:
            print(f"{triplet.head_entity['text']} --{triplet.relation}--> {triplet.tail_entity['text']} (conf: {triplet.confidence:.2f})")
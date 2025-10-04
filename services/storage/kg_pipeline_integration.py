#!/usr/bin/env python3
"""
KG Pipeline Integration

Integrates the ProductionKGStore with the existing annotation pipeline
to ensure all validated triplets are automatically stored in production.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from production_kg_store import ProductionKGStore, KGTriplet

logger = logging.getLogger(__name__)

class KGPipelineIntegration:
    """
    Integrates KG storage with annotation pipeline.
    
    Automatically processes gold annotations and stores them as KG triplets.
    """
    
    def __init__(self, 
                 kg_store_path: Path,
                 gold_store_path: Path,
                 enable_auto_processing: bool = True):
        """
        Initialize KG pipeline integration.
        
        Args:
            kg_store_path: Path to production KG store
            gold_store_path: Path to gold annotations
            enable_auto_processing: Enable automatic processing of new annotations
        """
        self.kg_store = ProductionKGStore(kg_store_path)
        self.gold_store_path = Path(gold_store_path)
        self.enable_auto_processing = enable_auto_processing
        
        # Track processed annotations to avoid duplicates
        self.processed_file = kg_store_path / "processed_annotations.json"
        self.processed_annotations = self._load_processed_annotations()
        
        logger.info(f"Initialized KG pipeline integration")
    
    def _load_processed_annotations(self) -> Dict[str, str]:
        """Load list of already processed annotation files"""
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_processed_annotations(self):
        """Save list of processed annotation files"""
        with open(self.processed_file, 'w') as f:
            json.dump(self.processed_annotations, f, indent=2)
    
    def process_annotation_file(self, annotation_file: Path) -> List[KGTriplet]:
        """
        Process a single gold annotation file into KG triplets.
        
        Args:
            annotation_file: Path to gold annotation JSON file
            
        Returns:
            List of created KG triplets
        """
        try:
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
            
            # Extract IDs from filename or annotation data
            doc_id = annotation_data.get("doc_id", annotation_file.stem.split('_')[0])
            sent_id = annotation_data.get("sent_id", annotation_file.stem.split('_')[1] if '_' in annotation_file.stem else "s0")
            
            # Store as KG triplets
            triplets = self.kg_store.store_gold_annotation(annotation_data, doc_id, sent_id)
            
            # Mark as processed
            file_hash = self._get_file_hash(annotation_file)
            self.processed_annotations[str(annotation_file)] = file_hash
            self._save_processed_annotations()
            
            logger.info(f"Processed {annotation_file.name}: {len(triplets)} triplets")
            return triplets
            
        except Exception as e:
            logger.error(f"Failed to process {annotation_file}: {e}")
            return []
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for change detection"""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def process_all_gold_annotations(self, force_reprocess: bool = False) -> int:
        """
        Process all gold annotation files.
        
        Args:
            force_reprocess: Reprocess files even if already processed
            
        Returns:
            Total number of triplets created
        """
        total_triplets = 0
        
        if not self.gold_store_path.exists():
            logger.warning(f"Gold store path does not exist: {self.gold_store_path}")
            return 0
        
        # Find all gold annotation files
        gold_files = list(self.gold_store_path.glob("*.json"))
        logger.info(f"Found {len(gold_files)} gold annotation files")
        
        for gold_file in gold_files:
            # Check if already processed
            file_path_str = str(gold_file)
            current_hash = self._get_file_hash(gold_file)
            
            if not force_reprocess and file_path_str in self.processed_annotations:
                if self.processed_annotations[file_path_str] == current_hash:
                    logger.debug(f"Skipping already processed file: {gold_file.name}")
                    continue
                else:
                    logger.info(f"File changed, reprocessing: {gold_file.name}")
            
            # Process the file
            triplets = self.process_annotation_file(gold_file)
            total_triplets += len(triplets)
        
        logger.info(f"Processed {len(gold_files)} files, created {total_triplets} total triplets")
        return total_triplets
    
    def setup_auto_processing(self):
        """
        Setup automatic processing of new gold annotations.
        
        This would typically be called by the annotation API when new
        annotations are validated and stored.
        """
        if not self.enable_auto_processing:
            logger.info("Auto-processing disabled")
            return
        
        # Process any existing unprocessed files
        self.process_all_gold_annotations()
        logger.info("Auto-processing setup complete")
    
    def validate_kg_completeness(self) -> Dict[str, Any]:
        """
        Validate that all gold annotations have been converted to KG triplets.
        
        Returns:
            Validation report with completeness statistics
        """
        # Count gold annotation files
        gold_files = list(self.gold_store_path.glob("*.json"))
        total_gold_files = len(gold_files)
        
        # Count relations in gold files
        total_gold_relations = 0
        files_with_relations = 0
        
        for gold_file in gold_files:
            try:
                with open(gold_file, 'r') as f:
                    data = json.load(f)
                relations = data.get("relations", [])
                if relations:
                    files_with_relations += 1
                    total_gold_relations += len(relations)
            except Exception as e:
                logger.warning(f"Failed to read {gold_file}: {e}")
        
        # Get KG store statistics
        kg_stats = self.kg_store.get_statistics()
        total_kg_triplets = kg_stats["total_triplets"]
        
        # Calculate completeness
        completeness_ratio = total_kg_triplets / total_gold_relations if total_gold_relations > 0 else 0
        
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "gold_annotations": {
                "total_files": total_gold_files,
                "files_with_relations": files_with_relations,
                "total_relations": total_gold_relations
            },
            "kg_store": {
                "total_triplets": total_kg_triplets,
                "completeness_ratio": completeness_ratio,
                "is_complete": completeness_ratio >= 0.95  # 95% threshold
            },
            "processed_files": len(self.processed_annotations),
            "recommendations": []
        }
        
        # Add recommendations
        if completeness_ratio < 0.95:
            validation_report["recommendations"].append(
                "Run process_all_gold_annotations() to ensure all annotations are converted to triplets"
            )
        
        if total_gold_files != len(self.processed_annotations):
            validation_report["recommendations"].append(
                f"Some files may not be processed: {total_gold_files} gold files vs {len(self.processed_annotations)} processed"
            )
        
        return validation_report
    
    def export_production_dataset(self, 
                                output_dir: Path,
                                formats: List[str] = ["jsonl", "csv", "rdf"],
                                min_confidence: float = 0.7) -> Dict[str, int]:
        """
        Export production-ready KG dataset in multiple formats.
        
        Args:
            output_dir: Output directory
            formats: List of export formats
            min_confidence: Minimum confidence threshold
            
        Returns:
            Export counts per format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_counts = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_name in formats:
            output_file = output_dir / f"shrimp_kg_{timestamp}.{format_name}"
            
            count = self.kg_store.export_dataset(
                output_file,
                format=format_name,
                min_confidence=min_confidence,
                include_metadata=True
            )
            
            export_counts[format_name] = count
            logger.info(f"Exported {count} triplets to {output_file}")
        
        # Also export statistics
        stats = self.kg_store.get_statistics()
        stats_file = output_dir / f"kg_statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Export validation report
        validation = self.validate_kg_completeness()
        validation_file = output_dir / f"kg_validation_{timestamp}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation, f, indent=2)
        
        return export_counts


def integrate_with_annotation_api():
    """
    Integration point for the annotation API.
    
    This should be called by the annotation API whenever a new
    gold annotation is created or updated.
    """
    # Initialize integration
    project_root = Path(__file__).parent.parent.parent
    kg_store_path = project_root / "data" / "production_kg"
    gold_store_path = project_root / "data" / "gold"
    
    integration = KGPipelineIntegration(kg_store_path, gold_store_path)
    
    # Process all annotations
    triplet_count = integration.process_all_gold_annotations()
    
    # Validate completeness
    validation = integration.validate_kg_completeness()
    
    logger.info(f"KG integration complete: {triplet_count} triplets, completeness: {validation['kg_store']['completeness_ratio']:.2%}")
    
    return validation


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KG Pipeline Integration")
    parser.add_argument("--kg-store-path", required=True, help="KG store base path")
    parser.add_argument("--gold-store-path", required=True, help="Gold annotations path")
    parser.add_argument("--action", choices=["process", "validate", "export"], 
                       default="process", help="Action to perform")
    parser.add_argument("--export-dir", help="Export directory")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum confidence for export")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing of all files")
    
    args = parser.parse_args()
    
    # Initialize integration
    integration = KGPipelineIntegration(
        Path(args.kg_store_path),
        Path(args.gold_store_path)
    )
    
    if args.action == "process":
        count = integration.process_all_gold_annotations(force_reprocess=args.force)
        print(f"Processed annotations: {count} triplets created")
    
    elif args.action == "validate":
        validation = integration.validate_kg_completeness()
        print(json.dumps(validation, indent=2))
    
    elif args.action == "export":
        if not args.export_dir:
            print("--export-dir required for export action")
            exit(1)
        
        counts = integration.export_production_dataset(
            Path(args.export_dir),
            min_confidence=args.min_confidence
        )
        print(f"Export complete: {counts}")
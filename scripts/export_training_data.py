#!/usr/bin/env python3
"""
Training Data Export Scripts

Exports gold annotations in various formats for ML training:
- SciBERT format (compatible with existing pipeline)
- CoNLL format for NER
- JSONL for relation extraction
- BIO format for token classification
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import argparse
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Training example with multiple format support"""
    text: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]
    doc_id: str
    sent_id: str
    metadata: Dict[str, Any]

class TrainingDataExporter:
    """
    Export gold annotations to various training formats.
    
    Supports multiple output formats and train/dev/test splits.
    """
    
    def __init__(self, 
                 gold_store_path: Path,
                 output_dir: Path,
                 data_training_path: Optional[Path] = None):
        """
        Initialize exporter.
        
        Args:
            gold_store_path: Path to gold annotations
            output_dir: Output directory for exports
            data_training_path: Path to original data-training project
        """
        self.gold_store_path = Path(gold_store_path)
        self.output_dir = Path(output_dir)
        self.data_training_path = Path(data_training_path) if data_training_path else None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load entity and relation mappings
        self._load_label_mappings()
        
        logger.info(f"Initialized exporter with gold store: {gold_store_path}")
    
    def _load_label_mappings(self):
        """Load label mappings for consistent export"""
        # Entity type mappings
        self.entity_labels = [
            "SPECIES", "PATHOGEN", "DISEASE", "GENE", "TREATMENT",
            "MEASUREMENT", "LIFE_STAGE", "MANAGEMENT_PRACTICE", 
            "ENVIRONMENTAL_PARAM", "LOCATION", "TRAIT"
        ]
        
        # Relation type mappings
        self.relation_labels = [
            "infected_by", "infects", "causes", "treated_with",
            "resistant_to", "associated_with", "measurement_of",
            "located_in", "affects_trait", "has_variant"
        ]
        
        # Topic mappings
        self.topic_labels = [
            "T_DISEASE", "T_TREATMENT", "T_GENETICS", "T_MANAGEMENT",
            "T_ENVIRONMENT", "T_REPRODUCTION", "T_NUTRITION",
            "T_BIOSECURITY", "T_ECONOMICS", "T_DIAGNOSTICS"
        ]
        
        # Create label-to-index mappings
        self.entity_to_idx = {label: idx for idx, label in enumerate(self.entity_labels)}
        self.relation_to_idx = {label: idx for idx, label in enumerate(self.relation_labels)}
        self.topic_to_idx = {label: idx for idx, label in enumerate(self.topic_labels)}
    
    def load_gold_annotations(self) -> List[TrainingExample]:
        """Load all gold annotations"""
        examples = []
        
        if not self.gold_store_path.exists():
            logger.error(f"Gold store path does not exist: {self.gold_store_path}")
            return examples
        
        # Load JSON files
        for gold_file in self.gold_store_path.glob("*.json"):
            try:
                with open(gold_file, 'r') as f:
                    gold_data = json.load(f)
                
                # Extract required fields
                text = gold_data.get("text", "")
                entities = gold_data.get("entities", [])
                relations = gold_data.get("relations", [])
                topics = gold_data.get("topics", [])
                doc_id = gold_data.get("doc_id", gold_file.stem)
                sent_id = gold_data.get("sent_id", "s0")
                
                # Create training example
                example = TrainingExample(
                    text=text,
                    entities=entities,
                    relations=relations,
                    topics=topics,
                    doc_id=doc_id,
                    sent_id=sent_id,
                    metadata={
                        "annotator": gold_data.get("annotator"),
                        "timestamp": gold_data.get("timestamp"),
                        "source_file": str(gold_file)
                    }
                )
                
                examples.append(example)
                
            except Exception as e:
                logger.error(f"Failed to load {gold_file}: {e}")
        
        logger.info(f"Loaded {len(examples)} gold annotations")
        return examples
    
    def create_splits(self, 
                     examples: List[TrainingExample],
                     train_ratio: float = 0.7,
                     dev_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_seed: int = 42) -> Tuple[List, List, List]:
        """
        Create train/dev/test splits.
        
        Args:
            examples: List of training examples
            train_ratio: Training set ratio
            dev_ratio: Development set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train, dev, test) lists
        """
        random.seed(random_seed)
        
        # Group by document to avoid data leakage
        doc_groups = defaultdict(list)
        for example in examples:
            doc_groups[example.doc_id].append(example)
        
        doc_ids = list(doc_groups.keys())
        random.shuffle(doc_ids)
        
        # Calculate split points
        n_docs = len(doc_ids)
        train_end = int(n_docs * train_ratio)
        dev_end = train_end + int(n_docs * dev_ratio)
        
        # Create splits
        train_docs = doc_ids[:train_end]
        dev_docs = doc_ids[train_end:dev_end]
        test_docs = doc_ids[dev_end:]
        
        # Collect examples
        train_examples = []
        dev_examples = []
        test_examples = []
        
        for doc_id in train_docs:
            train_examples.extend(doc_groups[doc_id])
        for doc_id in dev_docs:
            dev_examples.extend(doc_groups[doc_id])
        for doc_id in test_docs:
            test_examples.extend(doc_groups[doc_id])
        
        logger.info(f"Created splits: train={len(train_examples)}, dev={len(dev_examples)}, test={len(test_examples)}")
        return train_examples, dev_examples, test_examples
    
    def export_scibert_format(self, 
                            examples: List[TrainingExample],
                            output_file: Path):
        """
        Export in SciBERT format compatible with existing training pipeline.
        
        Format matches the existing data-training project structure.
        """
        scibert_data = []
        
        for example in examples:
            # Convert to SciBERT format
            scibert_item = {
                "text": example.text,
                "entities": [],
                "relations": [],
                "doc_id": example.doc_id,
                "sent_id": example.sent_id,
                "metadata": example.metadata
            }
            
            # Convert entities
            for entity in example.entities:
                scibert_entity = {
                    "id": entity.get("id", f"e{len(scibert_item['entities'])}"),
                    "text": entity["text"],
                    "label": entity["label"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "confidence": entity.get("confidence", 1.0)
                }
                
                # Add canonical form if available
                if "canonical" in entity:
                    scibert_entity["canonical"] = entity["canonical"]
                
                scibert_item["entities"].append(scibert_entity)
            
            # Convert relations
            for relation in example.relations:
                scibert_relation = {
                    "id": relation.get("id", f"r{len(scibert_item['relations'])}"),
                    "head": relation["head"],
                    "tail": relation["tail"],
                    "label": relation["label"],
                    "confidence": relation.get("confidence", 1.0)
                }
                
                # Add evidence span if available
                if "evidence_span" in relation:
                    scibert_relation["evidence_span"] = relation["evidence_span"]
                
                scibert_item["relations"].append(scibert_relation)
            
            scibert_data.append(scibert_item)
        
        # Save to JSONL format
        with open(output_file, 'w') as f:
            for item in scibert_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Exported {len(scibert_data)} examples to SciBERT format: {output_file}")
    
    def export_conll_format(self,
                          examples: List[TrainingExample],
                          output_file: Path):
        """
        Export in CoNLL format for NER training.
        
        Format: token\tlabel\n
        """
        with open(output_file, 'w') as f:
            for example in examples:
                # Tokenize text (simple whitespace tokenization)
                tokens = example.text.split()
                labels = ["O"] * len(tokens)
                
                # Create entity labels using BIO tagging
                for entity in example.entities:
                    entity_text = entity["text"]
                    entity_label = entity["label"]
                    
                    # Find entity tokens (simplified)
                    entity_tokens = entity_text.split()
                    for i, token in enumerate(tokens):
                        if token in entity_tokens:
                            if i == 0 or labels[i-1] == "O":
                                labels[i] = f"B-{entity_label}"
                            else:
                                labels[i] = f"I-{entity_label}"
                
                # Write tokens and labels
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                
                # Empty line between sentences
                f.write("\n")
        
        logger.info(f"Exported {len(examples)} examples to CoNLL format: {output_file}")
    
    def export_relation_format(self,
                              examples: List[TrainingExample],
                              output_file: Path):
        """
        Export in JSONL format for relation extraction training.
        """
        relation_data = []
        
        for example in examples:
            if not example.relations:
                continue
            
            for relation in example.relations:
                # Find head and tail entities
                head_entity = None
                tail_entity = None
                
                for entity in example.entities:
                    if entity.get("id") == relation["head"]:
                        head_entity = entity
                    if entity.get("id") == relation["tail"]:
                        tail_entity = entity
                
                if head_entity and tail_entity:
                    relation_item = {
                        "text": example.text,
                        "head": {
                            "text": head_entity["text"],
                            "label": head_entity["label"],
                            "start": head_entity["start"],
                            "end": head_entity["end"]
                        },
                        "tail": {
                            "text": tail_entity["text"],
                            "label": tail_entity["label"],
                            "start": tail_entity["start"],
                            "end": tail_entity["end"]
                        },
                        "relation": relation["label"],
                        "doc_id": example.doc_id,
                        "sent_id": example.sent_id
                    }
                    
                    relation_data.append(relation_item)
        
        # Save to JSONL format
        with open(output_file, 'w') as f:
            for item in relation_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Exported {len(relation_data)} relation examples: {output_file}")
    
    def export_topic_format(self,
                           examples: List[TrainingExample],
                           output_file: Path):
        """
        Export in format for topic classification training.
        """
        topic_data = []
        
        for example in examples:
            if example.topics:
                topic_item = {
                    "text": example.text,
                    "topics": [t["topic_id"] for t in example.topics],
                    "doc_id": example.doc_id,
                    "sent_id": example.sent_id
                }
                topic_data.append(topic_item)
        
        # Save to JSONL format
        with open(output_file, 'w') as f:
            for item in topic_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Exported {len(topic_data)} topic examples: {output_file}")
    
    def export_all_formats(self,
                          train_examples: List[TrainingExample],
                          dev_examples: List[TrainingExample],
                          test_examples: List[TrainingExample]):
        """Export all formats for all splits"""
        
        # Create format directories
        formats = ["scibert", "conll", "relation", "topic"]
        for format_name in formats:
            format_dir = self.output_dir / format_name
            format_dir.mkdir(exist_ok=True)
        
        splits = [
            ("train", train_examples),
            ("dev", dev_examples),
            ("test", test_examples)
        ]
        
        for split_name, examples in splits:
            if not examples:
                continue
            
            # SciBERT format
            self.export_scibert_format(
                examples,
                self.output_dir / "scibert" / f"{split_name}.jsonl"
            )
            
            # CoNLL format
            self.export_conll_format(
                examples,
                self.output_dir / "conll" / f"{split_name}.conll"
            )
            
            # Relation format
            self.export_relation_format(
                examples,
                self.output_dir / "relation" / f"{split_name}.jsonl"
            )
            
            # Topic format
            self.export_topic_format(
                examples,
                self.output_dir / "topic" / f"{split_name}.jsonl"
            )
    
    def create_label_mappings_file(self):
        """Create label mappings file for training scripts"""
        mappings = {
            "entity_labels": self.entity_labels,
            "relation_labels": self.relation_labels,
            "topic_labels": self.topic_labels,
            "entity_to_idx": self.entity_to_idx,
            "relation_to_idx": self.relation_to_idx,
            "topic_to_idx": self.topic_to_idx
        }
        
        with open(self.output_dir / "label_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info("Created label mappings file")
    
    def create_dataset_statistics(self, 
                                train_examples: List[TrainingExample],
                                dev_examples: List[TrainingExample],
                                test_examples: List[TrainingExample]):
        """Create dataset statistics file"""
        
        def analyze_split(examples, split_name):
            if not examples:
                return {}
            
            entity_counts = defaultdict(int)
            relation_counts = defaultdict(int)
            topic_counts = defaultdict(int)
            
            total_entities = 0
            total_relations = 0
            total_topics = 0
            
            for example in examples:
                for entity in example.entities:
                    entity_counts[entity["label"]] += 1
                    total_entities += 1
                
                for relation in example.relations:
                    relation_counts[relation["label"]] += 1
                    total_relations += 1
                
                for topic in example.topics:
                    topic_counts[topic["topic_id"]] += 1
                    total_topics += 1
            
            return {
                "examples": len(examples),
                "total_entities": total_entities,
                "total_relations": total_relations,
                "total_topics": total_topics,
                "avg_entities_per_example": total_entities / len(examples),
                "avg_relations_per_example": total_relations / len(examples),
                "avg_topics_per_example": total_topics / len(examples),
                "entity_distribution": dict(entity_counts),
                "relation_distribution": dict(relation_counts),
                "topic_distribution": dict(topic_counts)
            }
        
        statistics = {
            "generation_time": datetime.now().isoformat(),
            "total_examples": len(train_examples) + len(dev_examples) + len(test_examples),
            "splits": {
                "train": analyze_split(train_examples, "train"),
                "dev": analyze_split(dev_examples, "dev"),
                "test": analyze_split(test_examples, "test")
            }
        }
        
        with open(self.output_dir / "dataset_statistics.json", 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info("Created dataset statistics file")
    
    def export_complete_dataset(self,
                              train_ratio: float = 0.7,
                              dev_ratio: float = 0.15,
                              test_ratio: float = 0.15):
        """Export complete dataset with all formats and splits"""
        
        # Load gold annotations
        examples = self.load_gold_annotations()
        
        if not examples:
            logger.error("No gold annotations found")
            return
        
        # Create splits
        train_examples, dev_examples, test_examples = self.create_splits(
            examples, train_ratio, dev_ratio, test_ratio
        )
        
        # Export all formats
        self.export_all_formats(train_examples, dev_examples, test_examples)
        
        # Create metadata files
        self.create_label_mappings_file()
        self.create_dataset_statistics(train_examples, dev_examples, test_examples)
        
        logger.info(f"Complete dataset export finished: {self.output_dir}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Export training data from gold annotations")
    parser.add_argument("--gold-store", required=True, help="Path to gold annotations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--data-training-path", help="Path to data-training project")
    parser.add_argument("--format", choices=["all", "scibert", "conll", "relation", "topic"],
                       default="all", help="Export format")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.15, help="Dev set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = TrainingDataExporter(
        gold_store_path=Path(args.gold_store),
        output_dir=Path(args.output_dir),
        data_training_path=Path(args.data_training_path) if args.data_training_path else None
    )
    
    if args.format == "all":
        # Export complete dataset
        exporter.export_complete_dataset(
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio
        )
    else:
        # Export specific format
        examples = exporter.load_gold_annotations()
        train_examples, dev_examples, test_examples = exporter.create_splits(examples)
        
        output_file = Path(args.output_dir) / f"export.{args.format}"
        
        if args.format == "scibert":
            exporter.export_scibert_format(train_examples + dev_examples + test_examples, output_file)
        elif args.format == "conll":
            exporter.export_conll_format(train_examples + dev_examples + test_examples, output_file)
        elif args.format == "relation":
            exporter.export_relation_format(train_examples + dev_examples + test_examples, output_file)
        elif args.format == "topic":
            exporter.export_topic_format(train_examples + dev_examples + test_examples, output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

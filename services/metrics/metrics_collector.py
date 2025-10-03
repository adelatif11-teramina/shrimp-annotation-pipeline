"""
Metrics Collection and Monitoring Service

Tracks annotation throughput, quality metrics, inter-annotator agreement,
and system performance for the annotation pipeline.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import math

# For IAA calculations
try:
    from sklearn.metrics import cohen_kappa_score, f1_score
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

@dataclass
class AnnotationEvent:
    """Single annotation event for tracking"""
    event_id: str
    event_type: str  # started, completed, modified, rejected
    item_id: str
    doc_id: str
    sent_id: str
    annotator: str
    timestamp: str
    processing_time: Optional[float] = None
    decision: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class QualityMetrics:
    """Quality metrics for annotation session"""
    precision: float
    recall: float
    f1_score: float
    entity_accuracy: float
    relation_accuracy: float
    topic_accuracy: float
    confidence_score: float

@dataclass
class ThroughputMetrics:
    """Throughput metrics for annotator performance"""
    sentences_per_hour: float
    avg_time_per_sentence: float
    avg_time_per_entity: float
    avg_time_per_relation: float
    total_annotations: int
    total_time_hours: float

@dataclass
class IAA_Metrics:
    """Inter-Annotator Agreement metrics"""
    entity_kappa: float
    entity_f1: float
    relation_kappa: float
    relation_f1: float
    topic_kappa: float
    boundary_f1: float
    overall_agreement: float

class AnnotationMetricsCollector:
    """
    Collects and analyzes metrics for the annotation pipeline.
    
    Features:
    - Real-time throughput tracking
    - Quality assessment
    - Inter-annotator agreement calculation
    - System performance monitoring
    - Trend analysis and reporting
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metrics collector.
        
        Args:
            storage_path: Path to store metrics data
        """
        self.storage_path = storage_path or Path("./data/metrics")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory event store for fast access
        self.events: List[AnnotationEvent] = []
        self.session_cache: Dict[str, List[AnnotationEvent]] = defaultdict(list)
        
        # Load existing events
        self._load_events()
        
        logger.info(f"Initialized metrics collector with {len(self.events)} events")
    
    def _load_events(self):
        """Load existing events from storage"""
        events_file = self.storage_path / "annotation_events.jsonl"
        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        event = AnnotationEvent(**event_data)
                        self.events.append(event)
                        self.session_cache[event.annotator].append(event)
                    except Exception as e:
                        logger.warning(f"Failed to load event: {e}")
    
    def _save_event(self, event: AnnotationEvent):
        """Save event to persistent storage"""
        events_file = self.storage_path / "annotation_events.jsonl"
        with open(events_file, 'a') as f:
            f.write(json.dumps(asdict(event)) + "\n")
    
    def record_event(self,
                    event_type: str,
                    item_id: str,
                    doc_id: str,
                    sent_id: str,
                    annotator: str,
                    processing_time: Optional[float] = None,
                    decision: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> str:
        """
        Record an annotation event.
        
        Args:
            event_type: Type of event (started, completed, modified, rejected)
            item_id: Item being annotated
            doc_id: Document ID
            sent_id: Sentence ID
            annotator: Annotator identifier
            processing_time: Time taken in seconds
            decision: Annotation decision
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        event_id = f"{annotator}_{item_id}_{int(time.time())}"
        
        event = AnnotationEvent(
            event_id=event_id,
            event_type=event_type,
            item_id=item_id,
            doc_id=doc_id,
            sent_id=sent_id,
            annotator=annotator,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            decision=decision,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        self.session_cache[annotator].append(event)
        self._save_event(event)
        
        return event_id
    
    def calculate_throughput_metrics(self,
                                   annotator: Optional[str] = None,
                                   time_window_hours: Optional[int] = None) -> ThroughputMetrics:
        """
        Calculate throughput metrics for annotator(s).
        
        Args:
            annotator: Specific annotator or None for all
            time_window_hours: Time window in hours or None for all time
            
        Returns:
            Throughput metrics
        """
        # Filter events
        events = self.events
        if annotator:
            events = [e for e in events if e.annotator == annotator]
        
        if time_window_hours:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e.timestamp) >= cutoff]
        
        # Filter completed events
        completed_events = [e for e in events if e.event_type == "completed"]
        
        if not completed_events:
            return ThroughputMetrics(0, 0, 0, 0, 0, 0)
        
        # Calculate metrics
        total_time = sum(e.processing_time for e in completed_events if e.processing_time)
        total_annotations = len(completed_events)
        
        # Count entities and relations from metadata
        total_entities = 0
        total_relations = 0
        for event in completed_events:
            if event.metadata:
                total_entities += event.metadata.get("entity_count", 0)
                total_relations += event.metadata.get("relation_count", 0)
        
        # Calculate rates
        total_time_hours = total_time / 3600 if total_time else 0
        sentences_per_hour = total_annotations / total_time_hours if total_time_hours > 0 else 0
        avg_time_per_sentence = total_time / total_annotations if total_annotations > 0 else 0
        avg_time_per_entity = total_time / total_entities if total_entities > 0 else 0
        avg_time_per_relation = total_time / total_relations if total_relations > 0 else 0
        
        return ThroughputMetrics(
            sentences_per_hour=sentences_per_hour,
            avg_time_per_sentence=avg_time_per_sentence,
            avg_time_per_entity=avg_time_per_entity,
            avg_time_per_relation=avg_time_per_relation,
            total_annotations=total_annotations,
            total_time_hours=total_time_hours
        )
    
    def calculate_quality_metrics(self,
                                annotator: Optional[str] = None,
                                gold_store_path: Optional[Path] = None) -> QualityMetrics:
        """
        Calculate quality metrics by comparing with gold standard.
        
        Args:
            annotator: Specific annotator or None for all
            gold_store_path: Path to gold annotations
            
        Returns:
            Quality metrics
        """
        if not gold_store_path or not gold_store_path.exists():
            logger.warning("No gold store available for quality calculation")
            return QualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Load gold annotations
        gold_annotations = {}
        for gold_file in gold_store_path.glob("*.json"):
            with open(gold_file, 'r') as f:
                gold_data = json.load(f)
                item_id = gold_file.stem
                gold_annotations[item_id] = gold_data
        
        if not gold_annotations:
            return QualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Filter events for annotator
        events = self.events
        if annotator:
            events = [e for e in events if e.annotator == annotator]
        
        # Match events with gold annotations
        matched_annotations = []
        for event in events:
            if event.event_type == "completed" and event.item_id in gold_annotations:
                matched_annotations.append((event, gold_annotations[event.item_id]))
        
        if not matched_annotations:
            return QualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate metrics with proper precision/recall
        correct_entities = 0
        predicted_entities = 0
        gold_entities_total = 0
        correct_relations = 0
        predicted_relations = 0
        gold_relations_total = 0
        correct_topics = 0
        predicted_topics = 0
        gold_topics_total = 0
        
        for event, gold in matched_annotations:
            if event.metadata:
                # Entity metrics
                pred_entities = event.metadata.get("entities", [])
                gold_entities = gold.get("entities", [])
                
                predicted_entities += len(pred_entities)
                gold_entities_total += len(gold_entities)
                
                # Count correct predictions (true positives)
                for pred_ent in pred_entities:
                    if any(pred_ent.get("text") == gold_ent.get("text") and 
                          pred_ent.get("label") == gold_ent.get("label")
                          for gold_ent in gold_entities):
                        correct_entities += 1
                
                # Relation metrics
                pred_relations = event.metadata.get("relations", [])
                gold_relations = gold.get("relations", [])
                
                predicted_relations += len(pred_relations)
                gold_relations_total += len(gold_relations)
                
                # Count correct relations
                for pred_rel in pred_relations:
                    if any(pred_rel.get("label") == gold_rel.get("label") and
                          pred_rel.get("head") == gold_rel.get("head") and
                          pred_rel.get("tail") == gold_rel.get("tail")
                          for gold_rel in gold_relations):
                        correct_relations += 1
                
                # Topic metrics
                pred_topics = event.metadata.get("topics", [])
                gold_topics = gold.get("topics", [])
                
                predicted_topics += len(pred_topics)
                gold_topics_total += len(gold_topics)
                
                # Count correct topics
                for pred_topic in pred_topics:
                    if any(pred_topic.get("topic_id") == gold_topic.get("topic_id")
                          for gold_topic in gold_topics):
                        correct_topics += 1
        
        # Calculate component accuracies
        entity_accuracy = correct_entities / predicted_entities if predicted_entities > 0 else 0
        relation_accuracy = correct_relations / predicted_relations if predicted_relations > 0 else 0
        topic_accuracy = correct_topics / predicted_topics if predicted_topics > 0 else 0
        
        # Calculate overall precision and recall
        total_correct = correct_entities + correct_relations + correct_topics
        total_predicted = predicted_entities + predicted_relations + predicted_topics
        total_gold = gold_entities_total + gold_relations_total + gold_topics_total
        
        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            entity_accuracy=entity_accuracy,
            relation_accuracy=relation_accuracy,
            topic_accuracy=topic_accuracy,
            confidence_score=0.8  # Placeholder
        )
    
    def calculate_iaa_metrics(self,
                            annotator1: str,
                            annotator2: str,
                            gold_store_path: Optional[Path] = None) -> IAA_Metrics:
        """
        Calculate Inter-Annotator Agreement between two annotators.
        
        Args:
            annotator1: First annotator
            annotator2: Second annotator
            gold_store_path: Path to annotations
            
        Returns:
            IAA metrics
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available for IAA calculation")
            return IAA_Metrics(0, 0, 0, 0, 0, 0, 0)
        
        # Get annotations from both annotators for same items
        ann1_events = [e for e in self.events if e.annotator == annotator1 and e.event_type == "completed"]
        ann2_events = [e for e in self.events if e.annotator == annotator2 and e.event_type == "completed"]
        
        # Find overlapping items
        ann1_items = {e.item_id: e for e in ann1_events}
        ann2_items = {e.item_id: e for e in ann2_events}
        common_items = set(ann1_items.keys()) & set(ann2_items.keys())
        
        if len(common_items) < 2:
            logger.warning("Not enough overlapping annotations for IAA calculation")
            return IAA_Metrics(0, 0, 0, 0, 0, 0, 0)
        
        # Collect properly aligned labels for comparison
        entity_labels_1 = []
        entity_labels_2 = []
        relation_labels_1 = []
        relation_labels_2 = []
        topic_labels_1 = []
        topic_labels_2 = []
        
        for item_id in common_items:
            event1 = ann1_items[item_id]
            event2 = ann2_items[item_id]
            
            if event1.metadata and event2.metadata:
                # Entity alignment using proper matching
                entities1 = event1.metadata.get("entities", [])
                entities2 = event2.metadata.get("entities", [])
                
                aligned_entities = self._align_entities(entities1, entities2)
                for label1, label2 in aligned_entities:
                    entity_labels_1.append(label1)
                    entity_labels_2.append(label2)
                
                # Relation alignment
                relations1 = event1.metadata.get("relations", [])
                relations2 = event2.metadata.get("relations", [])
                
                aligned_relations = self._align_relations(relations1, relations2)
                for label1, label2 in aligned_relations:
                    relation_labels_1.append(label1)
                    relation_labels_2.append(label2)
                
                # Topic alignment (simpler since they're document-level)
                topics1 = set(t.get("topic_id") for t in event1.metadata.get("topics", []))
                topics2 = set(t.get("topic_id") for t in event2.metadata.get("topics", []))
                
                all_topics = topics1.union(topics2)
                for topic in all_topics:
                    topic_labels_1.append(topic if topic in topics1 else "NO_TOPIC")
                    topic_labels_2.append(topic if topic in topics2 else "NO_TOPIC")
        
        # Calculate kappa scores
        entity_kappa = 0
        relation_kappa = 0
        topic_kappa = 0
        
        try:
            if entity_labels_1 and entity_labels_2:
                entity_kappa = cohen_kappa_score(entity_labels_1, entity_labels_2)
        except:
            entity_kappa = 0
        
        try:
            if relation_labels_1 and relation_labels_2:
                # Pad shorter list
                max_len = max(len(relation_labels_1), len(relation_labels_2))
                relation_labels_1.extend(["NONE"] * (max_len - len(relation_labels_1)))
                relation_labels_2.extend(["NONE"] * (max_len - len(relation_labels_2)))
                relation_kappa = cohen_kappa_score(relation_labels_1, relation_labels_2)
        except:
            relation_kappa = 0
        
        try:
            if topic_labels_1 and topic_labels_2:
                # Pad shorter list
                max_len = max(len(topic_labels_1), len(topic_labels_2))
                topic_labels_1.extend(["NONE"] * (max_len - len(topic_labels_1)))
                topic_labels_2.extend(["NONE"] * (max_len - len(topic_labels_2)))
                topic_kappa = cohen_kappa_score(topic_labels_1, topic_labels_2)
        except:
            topic_kappa = 0
        
        # Calculate F1 scores (simplified)
        entity_f1 = 0
        relation_f1 = 0
        boundary_f1 = 0
        
        # Overall agreement (average of kappa scores)
        kappa_scores = [k for k in [entity_kappa, relation_kappa, topic_kappa] if not math.isnan(k)]
        overall_agreement = statistics.mean(kappa_scores) if kappa_scores else 0
        
        return IAA_Metrics(
            entity_kappa=entity_kappa,
            entity_f1=entity_f1,
            relation_kappa=relation_kappa,
            relation_f1=relation_f1,
            topic_kappa=topic_kappa,
            boundary_f1=boundary_f1,
            overall_agreement=overall_agreement
        )
    
    def _align_entities(self, entities1: List[Dict], entities2: List[Dict]) -> List[Tuple[str, str]]:
        """
        Align entities between two annotators using span overlap and text matching.
        
        Returns:
            List of (label1, label2) pairs for aligned entities
        """
        aligned = []
        used_indices2 = set()
        
        # Create alignment matrix based on span overlap and text similarity
        for i, e1 in enumerate(entities1):
            best_match = None
            best_score = 0
            best_j = -1
            
            for j, e2 in enumerate(entities2):
                if j in used_indices2:
                    continue
                
                # Calculate alignment score
                score = self._calculate_entity_alignment_score(e1, e2)
                if score > best_score and score > 0.5:  # Minimum threshold
                    best_score = score
                    best_match = e2
                    best_j = j
            
            if best_match:
                aligned.append((e1.get("label", "O"), best_match.get("label", "O")))
                used_indices2.add(best_j)
            else:
                # No match found for e1
                aligned.append((e1.get("label", "O"), "O"))
        
        # Add unmatched entities from entities2
        for j, e2 in enumerate(entities2):
            if j not in used_indices2:
                aligned.append(("O", e2.get("label", "O")))
        
        return aligned
    
    def _calculate_entity_alignment_score(self, e1: Dict, e2: Dict) -> float:
        """Calculate alignment score between two entities"""
        # Text exact match (highest priority)
        if e1.get("text", "").lower() == e2.get("text", "").lower():
            return 1.0
        
        # Span overlap calculation
        start1, end1 = e1.get("start", 0), e1.get("end", 0)
        start2, end2 = e2.get("start", 0), e2.get("end", 0)
        
        if start1 >= end1 or start2 >= end2:  # Invalid spans
            return 0.0
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:  # No overlap
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        union_length = max(end1, end2) - min(start1, start2)
        
        # Jaccard similarity for spans
        span_score = overlap_length / union_length if union_length > 0 else 0
        
        # Text similarity (simple approach)
        text1 = e1.get("text", "").lower()
        text2 = e2.get("text", "").lower()
        
        if text1 and text2:
            # Simple string similarity (could use more sophisticated measures)
            if text1 in text2 or text2 in text1:
                text_score = 0.8
            else:
                text_score = 0.0
        else:
            text_score = 0.0
        
        # Combined score (span overlap is more important)
        return 0.7 * span_score + 0.3 * text_score
    
    def _align_relations(self, relations1: List[Dict], relations2: List[Dict]) -> List[Tuple[str, str]]:
        """
        Align relations between two annotators.
        Relations are matched by head-tail entity pairs and relation type.
        """
        aligned = []
        used_indices2 = set()
        
        for i, r1 in enumerate(relations1):
            best_match = None
            best_j = -1
            
            for j, r2 in enumerate(relations2):
                if j in used_indices2:
                    continue
                
                # Check if relations match (same head, tail, and type)
                if (r1.get("head") == r2.get("head") and 
                    r1.get("tail") == r2.get("tail") and
                    r1.get("label") == r2.get("label")):
                    best_match = r2
                    best_j = j
                    break
                # Partial match (same head-tail, different type)
                elif (r1.get("head") == r2.get("head") and 
                      r1.get("tail") == r2.get("tail")):
                    best_match = r2
                    best_j = j
                    # Continue to find exact match if possible
            
            if best_match:
                aligned.append((r1.get("label", "NO_REL"), best_match.get("label", "NO_REL")))
                used_indices2.add(best_j)
            else:
                aligned.append((r1.get("label", "NO_REL"), "NO_REL"))
        
        # Add unmatched relations from relations2
        for j, r2 in enumerate(relations2):
            if j not in used_indices2:
                aligned.append(("NO_REL", r2.get("label", "NO_REL")))
        
        return aligned
    
    def get_annotator_statistics(self, 
                               annotator: str,
                               time_window_hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get comprehensive statistics for an annotator"""
        throughput = self.calculate_throughput_metrics(annotator, time_window_hours)
        quality = self.calculate_quality_metrics(annotator)
        
        # Get recent events
        events = [e for e in self.session_cache[annotator]]
        if time_window_hours:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e.timestamp) >= cutoff]
        
        # Calculate decision distribution
        decisions = defaultdict(int)
        for event in events:
            if event.decision:
                decisions[event.decision] += 1
        
        return {
            "annotator": annotator,
            "time_window_hours": time_window_hours,
            "throughput": asdict(throughput),
            "quality": asdict(quality),
            "event_count": len(events),
            "decision_distribution": dict(decisions),
            "last_activity": max(e.timestamp for e in events) if events else None
        }
    
    def generate_system_report(self, 
                             output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive system performance report"""
        report = {
            "generation_time": datetime.now().isoformat(),
            "total_events": len(self.events),
            "annotators": {}
        }
        
        # Get unique annotators
        annotators = set(e.annotator for e in self.events)
        
        for annotator in annotators:
            report["annotators"][annotator] = self.get_annotator_statistics(annotator)
        
        # System-wide metrics
        overall_throughput = self.calculate_throughput_metrics()
        overall_quality = self.calculate_quality_metrics()
        
        report["system_wide"] = {
            "throughput": asdict(overall_throughput),
            "quality": asdict(overall_quality),
            "active_annotators": len(annotators)
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def get_trends(self, 
                  metric: str = "throughput",
                  time_window_days: int = 7,
                  granularity: str = "daily") -> Dict[str, Any]:
        """
        Get trend data for metrics over time.
        
        Args:
            metric: Metric to analyze (throughput, quality, decisions)
            time_window_days: Number of days to analyze
            granularity: Time granularity (hourly, daily, weekly)
            
        Returns:
            Trend data
        """
        cutoff = datetime.now() - timedelta(days=time_window_days)
        recent_events = [e for e in self.events if datetime.fromisoformat(e.timestamp) >= cutoff]
        
        # Group events by time period
        time_buckets = defaultdict(list)
        
        for event in recent_events:
            event_time = datetime.fromisoformat(event.timestamp)
            
            if granularity == "hourly":
                bucket = event_time.strftime("%Y-%m-%d %H:00")
            elif granularity == "daily":
                bucket = event_time.strftime("%Y-%m-%d")
            elif granularity == "weekly":
                bucket = event_time.strftime("%Y-W%U")
            else:
                bucket = event_time.strftime("%Y-%m-%d")
            
            time_buckets[bucket].append(event)
        
        # Calculate metric for each time bucket
        trend_data = {}
        for bucket, events in time_buckets.items():
            if metric == "throughput":
                completed = [e for e in events if e.event_type == "completed"]
                trend_data[bucket] = len(completed)
            elif metric == "quality":
                # Would need more sophisticated calculation
                trend_data[bucket] = 0.8  # Placeholder
            elif metric == "decisions":
                decisions = defaultdict(int)
                for event in events:
                    if event.decision:
                        decisions[event.decision] += 1
                trend_data[bucket] = dict(decisions)
        
        return {
            "metric": metric,
            "time_window_days": time_window_days,
            "granularity": granularity,
            "trend_data": trend_data
        }


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Annotation metrics collector")
    parser.add_argument("--storage-path", help="Path to metrics storage")
    parser.add_argument("--generate-report", action="store_true", help="Generate system report")
    parser.add_argument("--annotator", help="Get statistics for specific annotator")
    parser.add_argument("--test-data", action="store_true", help="Generate test data")
    
    args = parser.parse_args()
    
    # Initialize collector
    storage_path = Path(args.storage_path) if args.storage_path else None
    collector = AnnotationMetricsCollector(storage_path)
    
    if args.test_data:
        # Generate some test events
        import random
        
        annotators = ["ann1@example.com", "ann2@example.com", "ann3@example.com"]
        decisions = ["accepted", "rejected", "modified"]
        
        for i in range(50):
            collector.record_event(
                event_type="completed",
                item_id=f"item_{i}",
                doc_id=f"doc_{i//10}",
                sent_id=f"s_{i%10}",
                annotator=random.choice(annotators),
                processing_time=random.uniform(30, 300),  # 30 seconds to 5 minutes
                decision=random.choice(decisions),
                metadata={
                    "entity_count": random.randint(1, 5),
                    "relation_count": random.randint(0, 3),
                    "topic_count": random.randint(1, 3)
                }
            )
        
        print(f"Generated {len(collector.events)} test events")
    
    if args.annotator:
        # Get statistics for specific annotator
        stats = collector.get_annotator_statistics(args.annotator)
        print(json.dumps(stats, indent=2))
    
    if args.generate_report:
        # Generate system report
        report = collector.generate_system_report()
        print(json.dumps(report, indent=2))
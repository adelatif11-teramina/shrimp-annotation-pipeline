"""
Triage and Prioritization Service

Intelligently prioritizes annotation candidates for human review based on
confidence, novelty, impact, and disagreement between models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import heapq
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class PriorityLevel(Enum):
    """Priority levels for triage"""
    CRITICAL = 1  # Must review immediately
    HIGH = 2      # Important to review
    MEDIUM = 3    # Standard priority
    LOW = 4       # Can wait
    MINIMAL = 5   # Review if time permits

@dataclass
class TriageItem:
    """Item in the triage queue"""
    item_id: str
    doc_id: str
    sent_id: str
    item_type: str  # entity, relation, topic
    priority_score: float
    priority_level: PriorityLevel
    
    # Scoring components
    confidence_score: float = 0.0
    novelty_score: float = 0.0
    impact_score: float = 0.0
    disagreement_score: float = 0.0
    authority_score: float = 0.0
    
    # Metadata
    candidate_data: Dict = field(default_factory=dict)
    rule_data: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, in_review, completed, skipped
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority_score > other.priority_score

class TriagePrioritizationEngine:
    """
    Engine for prioritizing annotation candidates.
    
    Features:
    - Multi-factor scoring (confidence, novelty, impact, disagreement)
    - Source authority weighting
    - Dynamic priority adjustment
    - Queue management
    - Batch assignment
    """
    
    def __init__(self,
                 config: Optional[Dict] = None,
                 gold_store_path: Optional[Path] = None):
        """
        Initialize the triage engine.
        
        Args:
            config: Configuration for scoring weights
            gold_store_path: Path to gold annotation store
        """
        # Default scoring weights
        self.config = config or {
            "weights": {
                "confidence": 0.2,
                "novelty": 0.3,
                "impact": 0.25,
                "disagreement": 0.15,
                "authority": 0.1
            },
            "thresholds": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.5,
                "low": 0.3
            },
            "source_authority": {
                "paper": 1.0,
                "report": 0.9,
                "manual": 0.8,
                "hatchery_log": 0.6,
                "dataset": 0.7
            }
        }
        
        # Priority queue
        self.queue: List[TriageItem] = []
        
        # Tracking structures
        self.seen_entities: Set[str] = set()
        self.seen_relations: Set[Tuple[str, str, str]] = set()
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Load existing gold annotations if available
        self.gold_store_path = Path(gold_store_path) if gold_store_path else None
        self._load_gold_annotations()
        
        logger.info("Initialized triage engine")
    
    def _load_gold_annotations(self):
        """Load existing gold annotations to track what's been seen"""
        if not self.gold_store_path or not self.gold_store_path.exists():
            return
        
        gold_files = self.gold_store_path.glob("*.json")
        for file_path in gold_files:
            try:
                with open(file_path, 'r') as f:
                    gold = json.load(f)
                    
                    # Track seen entities
                    for entity in gold.get("entities", []):
                        canonical = entity.get("canonical", entity["text"])
                        self.seen_entities.add(canonical.lower())
                    
                    # Track seen relations
                    for relation in gold.get("relations", []):
                        # Note: would need entity resolution here
                        pass
                        
            except Exception as e:
                logger.warning(f"Failed to load gold file {file_path}: {e}")
    
    def calculate_confidence_score(self, candidate: Dict) -> float:
        """
        Calculate confidence-based score.
        High confidence with disagreement = high priority.
        Low confidence = high priority for verification.
        """
        llm_confidence = candidate.get("confidence", 0.5)
        
        # U-shaped curve: prioritize very high and very low confidence
        if llm_confidence > 0.8:
            # High confidence - verify if correct
            return 0.7 + (llm_confidence - 0.8) * 1.5
        elif llm_confidence < 0.3:
            # Low confidence - needs human decision
            return 0.8 - llm_confidence
        else:
            # Medium confidence - lower priority
            return 0.3 + (llm_confidence - 0.3) * 0.4
    
    def calculate_novelty_score(self, candidate: Dict, item_type: str) -> float:
        """
        Calculate novelty score.
        Novel entities/relations get higher priority.
        """
        if item_type == "entity":
            text = candidate.get("text", "").lower()
            canonical = candidate.get("canonical", text).lower()
            
            # Check if seen before
            if canonical in self.seen_entities:
                return 0.1
            
            # Check for similar entities (fuzzy matching)
            for seen in self.seen_entities:
                if text in seen or seen in text:
                    return 0.3
            
            # Completely novel
            return 0.9
            
        elif item_type == "relation":
            # Check relation novelty
            head = candidate.get("head_text", "").lower()
            tail = candidate.get("tail_text", "").lower()
            label = candidate.get("label", "")
            
            rel_tuple = (head, label, tail)
            if rel_tuple in self.seen_relations:
                return 0.1
            
            # Check if entities are connected in graph
            if head in self.entity_graph and tail in self.entity_graph[head]:
                return 0.4
            
            # Novel connection
            return 0.85
            
        elif item_type == "topic":
            # Topics are generally not novel after initial setup
            return 0.3
        
        return 0.5
    
    def calculate_impact_score(self, candidate: Dict, item_type: str) -> float:
        """
        Calculate potential impact on knowledge graph.
        High-impact terms and relations get priority.
        """
        impact_terms = {
            "disease": ["ahpnd", "tpd", "wsd", "ems", "mortality", "outbreak"],
            "pathogen": ["vibrio", "virus", "bacteria", "parasite", "ehp"],
            "treatment": ["antibiotic", "probiotic", "vaccine", "therapy"],
            "resistance": ["resistant", "susceptible", "tolerance"],
            "economic": ["loss", "cost", "profit", "yield"]
        }
        
        if item_type == "entity":
            text = candidate.get("text", "").lower()
            label = candidate.get("label", "")
            
            # Check for high-impact entity types
            if label in ["PATHOGEN", "DISEASE", "TREATMENT"]:
                base_score = 0.7
            else:
                base_score = 0.4
            
            # Check for impact terms
            for category, terms in impact_terms.items():
                if any(term in text for term in terms):
                    base_score += 0.2
                    break
            
            return min(base_score, 1.0)
            
        elif item_type == "relation":
            label = candidate.get("label", "")
            
            # High-impact relation types
            if label in ["causes", "infected_by", "resistant_to", "treated_with"]:
                return 0.8
            elif label in ["associated_with", "affects_trait"]:
                return 0.6
            else:
                return 0.4
                
        return 0.5
    
    def calculate_disagreement_score(self, 
                                    candidate: Dict,
                                    rule_result: Optional[Dict]) -> float:
        """
        Calculate disagreement between LLM and rule-based systems.
        High disagreement = high priority for human resolution.
        """
        if not rule_result:
            # No rule result to compare
            return 0.3
        
        # Check for conflicts
        llm_label = candidate.get("label", "")
        rule_label = rule_result.get("label", "")
        
        # Check if both labels are actually present (non-empty)
        both_labels_present = bool(llm_label and rule_label)
        
        if llm_label != rule_label and both_labels_present:
            # Direct conflict
            return 0.9
        
        # Check confidence disagreement
        llm_conf = candidate.get("confidence", 0.5)
        rule_conf = rule_result.get("confidence", 0.5)
        conf_diff = abs(llm_conf - rule_conf)
        
        if conf_diff > 0.5:
            return 0.7
        
        return conf_diff
    
    def calculate_authority_score(self, source: str) -> float:
        """Calculate source authority score"""
        return self.config["source_authority"].get(source, 0.5)
    
    def calculate_priority(self,
                          candidate: Dict,
                          doc_metadata: Dict,
                          rule_result: Optional[Dict] = None,
                          item_type: str = "entity") -> TriageItem:
        """
        Calculate overall priority for a candidate.
        
        Args:
            candidate: Candidate data
            doc_metadata: Document metadata
            rule_result: Optional rule-based result
            item_type: Type of item (entity, relation, topic)
            
        Returns:
            TriageItem with calculated priority
        """
        # Calculate component scores
        confidence = self.calculate_confidence_score(candidate)
        novelty = self.calculate_novelty_score(candidate, item_type)
        impact = self.calculate_impact_score(candidate, item_type)
        disagreement = self.calculate_disagreement_score(candidate, rule_result)
        authority = self.calculate_authority_score(doc_metadata.get("source", "unknown"))
        
        # Calculate weighted priority score
        weights = self.config["weights"]
        priority_score = (
            weights["confidence"] * confidence +
            weights["novelty"] * novelty +
            weights["impact"] * impact +
            weights["disagreement"] * disagreement +
            weights["authority"] * authority
        )
        
        # Determine priority level
        thresholds = self.config["thresholds"]
        if priority_score >= thresholds["critical"]:
            priority_level = PriorityLevel.CRITICAL
        elif priority_score >= thresholds["high"]:
            priority_level = PriorityLevel.HIGH
        elif priority_score >= thresholds["medium"]:
            priority_level = PriorityLevel.MEDIUM
        elif priority_score >= thresholds["low"]:
            priority_level = PriorityLevel.LOW
        else:
            priority_level = PriorityLevel.MINIMAL
        
        # Create triage item
        item_id = f"{doc_metadata['doc_id']}_{doc_metadata['sent_id']}_{item_type}_{len(self.queue)}"
        
        return TriageItem(
            item_id=item_id,
            doc_id=doc_metadata["doc_id"],
            sent_id=doc_metadata["sent_id"],
            item_type=item_type,
            priority_score=priority_score,
            priority_level=priority_level,
            confidence_score=confidence,
            novelty_score=novelty,
            impact_score=impact,
            disagreement_score=disagreement,
            authority_score=authority,
            candidate_data=candidate,
            rule_data=rule_result
        )
    
    def add_candidates(self,
                      candidates: Dict[str, Any],
                      doc_metadata: Dict,
                      rule_results: Optional[Dict] = None):
        """
        Add candidates to triage queue.
        
        Args:
            candidates: Dictionary with entities, relations, topics
            doc_metadata: Document metadata
            rule_results: Optional rule-based results
        """
        items = []
        
        # Process entities
        for entity in candidates.get("entities", []):
            rule_entity = None
            if rule_results:
                # Match rule entity by text/position
                for re in rule_results.get("entities", []):
                    if re["start"] == entity["start"] and re["end"] == entity["end"]:
                        rule_entity = re
                        break
            
            item = self.calculate_priority(
                entity, doc_metadata, rule_entity, "entity"
            )
            items.append(item)
        
        # Process relations
        for relation in candidates.get("relations", []):
            rule_relation = None
            if rule_results:
                # Match rule relation
                for rr in rule_results.get("relations", []):
                    if (rr.get("head_cid") == relation.get("head_cid") and
                        rr.get("tail_cid") == relation.get("tail_cid")):
                        rule_relation = rr
                        break
            
            item = self.calculate_priority(
                relation, doc_metadata, rule_relation, "relation"
            )
            items.append(item)
        
        # Process topics
        for topic in candidates.get("topics", []):
            item = self.calculate_priority(
                topic, doc_metadata, None, "topic"
            )
            items.append(item)
        
        # Add to queue
        for item in items:
            heapq.heappush(self.queue, item)
        
        logger.info(f"Added {len(items)} items to triage queue")
    
    def get_next_batch(self,
                       batch_size: int = 10,
                       annotator: Optional[str] = None) -> List[TriageItem]:
        """
        Get next batch of items for review.
        
        Args:
            batch_size: Number of items to return
            annotator: Optional annotator to assign to
            
        Returns:
            List of highest priority items
        """
        batch = []
        
        # Get items from priority queue
        while len(batch) < batch_size and self.queue:
            item = heapq.heappop(self.queue)
            
            # Skip completed items
            if item.status == "completed":
                continue
            
            # Assign annotator
            if annotator:
                item.assigned_to = annotator
                item.status = "in_review"
            
            batch.append(item)
        
        # Re-add unassigned items
        for item in batch:
            if not annotator:
                heapq.heappush(self.queue, item)
        
        return batch
    
    def mark_completed(self, item_id: str, decision: str):
        """
        Mark an item as completed.
        
        Args:
            item_id: Item ID
            decision: Decision made (accepted, rejected, modified)
        """
        # Find and update item
        for i, item in enumerate(self.queue):
            if item.item_id == item_id:
                item.status = "completed"
                
                # Update tracking structures
                if decision == "accepted" and item.item_type == "entity":
                    text = item.candidate_data.get("text", "").lower()
                    self.seen_entities.add(text)
                
                break
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the triage queue"""
        if not self.queue:
            return {
                "total_items": 0,
                "by_priority": {},
                "by_type": {},
                "by_status": {}
            }
        
        stats = {
            "total_items": len(self.queue),
            "by_priority": defaultdict(int),
            "by_type": defaultdict(int),
            "by_status": defaultdict(int),
            "avg_priority_score": np.mean([item.priority_score for item in self.queue]),
            "pending_critical": 0,
            "pending_high": 0
        }
        
        for item in self.queue:
            stats["by_priority"][item.priority_level.name] += 1
            stats["by_type"][item.item_type] += 1
            stats["by_status"][item.status] += 1
            
            if item.status == "pending":
                if item.priority_level == PriorityLevel.CRITICAL:
                    stats["pending_critical"] += 1
                elif item.priority_level == PriorityLevel.HIGH:
                    stats["pending_high"] += 1
        
        return dict(stats)
    
    def export_queue(self, output_path: Path):
        """Export queue to file for persistence"""
        queue_data = [asdict(item) for item in self.queue]
        with open(output_path, 'w') as f:
            json.dump(queue_data, f, indent=2, default=str)
    
    def import_queue(self, input_path: Path):
        """Import queue from file"""
        with open(input_path, 'r') as f:
            queue_data = json.load(f)
        
        self.queue = []
        for item_data in queue_data:
            # Convert priority level
            item_data["priority_level"] = PriorityLevel[item_data["priority_level"]]
            item = TriageItem(**item_data)
            heapq.heappush(self.queue, item)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Triage prioritization engine")
    parser.add_argument("--test", action="store_true", help="Run test example")
    parser.add_argument("--stats", help="Get statistics for queue file")
    
    args = parser.parse_args()
    
    if args.test:
        # Create test engine
        engine = TriagePrioritizationEngine()
        
        # Test candidates
        test_candidates = {
            "entities": [
                {"text": "Vibrio parahaemolyticus", "label": "PATHOGEN", 
                 "start": 0, "end": 23, "confidence": 0.95},
                {"text": "AHPND", "label": "DISEASE",
                 "start": 30, "end": 35, "confidence": 0.9}
            ],
            "relations": [
                {"head_cid": 0, "tail_cid": 1, "label": "causes",
                 "confidence": 0.85}
            ],
            "topics": [
                {"topic_id": "T_DISEASE", "score": 0.8}
            ]
        }
        
        # Add to queue
        engine.add_candidates(
            test_candidates,
            {"doc_id": "test_doc", "sent_id": "s1", "source": "paper"}
        )
        
        # Get batch
        batch = engine.get_next_batch(5)
        
        print("Triage Queue (Top 5):")
        for item in batch:
            print(f"  {item.priority_level.name}: {item.item_type} - {item.candidate_data.get('text', item.candidate_data.get('label', 'N/A'))}")
            print(f"    Priority: {item.priority_score:.3f} (C:{item.confidence_score:.2f} N:{item.novelty_score:.2f} I:{item.impact_score:.2f})")
        
        # Get statistics
        stats = engine.get_queue_statistics()
        print(f"\nQueue Statistics:")
        print(f"  Total items: {stats['total_items']}")
        print(f"  By priority: {dict(stats['by_priority'])}")
        print(f"  By type: {dict(stats['by_type'])}")
    
    elif args.stats:
        # Load and analyze queue
        engine = TriagePrioritizationEngine()
        engine.import_queue(Path(args.stats))
        stats = engine.get_queue_statistics()
        print(json.dumps(stats, indent=2))
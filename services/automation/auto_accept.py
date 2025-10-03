"""
Auto-Accept Threshold System

Automatically accepts high-confidence annotations that meet quality thresholds,
reducing human annotation workload while maintaining quality standards.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class AutoAcceptRule:
    """Rule for automatic acceptance"""
    rule_id: str
    rule_name: str
    entity_types: Optional[List[str]] = None  # Applicable entity types
    relation_types: Optional[List[str]] = None  # Applicable relation types
    min_confidence: float = 0.95  # Minimum confidence threshold
    min_agreement: float = 0.9  # Minimum LLM-rule agreement
    source_authority_min: float = 0.8  # Minimum source authority
    requires_rule_support: bool = True  # Must have rule-based support
    max_novelty: float = 0.3  # Maximum novelty score (prefer known patterns)
    enabled: bool = True
    
    # Performance tracking
    auto_accepted: int = 0
    false_positives: int = 0
    precision: float = 1.0

@dataclass 
class AutoAcceptDecision:
    """Decision from auto-accept system"""
    item_id: str
    decision: str  # auto_accept, human_review, auto_reject
    confidence: float
    applied_rules: List[str]
    reasoning: str
    timestamp: str

class AutoAcceptSystem:
    """
    System for automatically accepting high-quality annotations.
    
    Features:
    - Configurable confidence thresholds
    - Multi-factor decision making
    - Performance tracking and rule adjustment
    - Safe fallback to human review
    - Audit trail for all decisions
    """
    
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 gold_store_path: Optional[Path] = None):
        """
        Initialize auto-accept system.
        
        Args:
            config_path: Path to configuration file
            gold_store_path: Path to gold annotations for validation
        """
        self.config_path = config_path
        self.gold_store_path = Path(gold_store_path) if gold_store_path else None
        
        # Load configuration
        self._load_config()
        
        # Performance tracking
        self.decisions: List[AutoAcceptDecision] = []
        self.performance_history = defaultdict(list)
        
        # Load decision history
        self._load_decision_history()
        
        logger.info(f"Initialized auto-accept system with {len(self.rules)} rules")
    
    def _load_config(self):
        """Load auto-accept configuration and rules"""
        default_rules = [
            # High-confidence entity rules
            AutoAcceptRule(
                rule_id="species_high_conf",
                rule_name="High-confidence species annotations",
                entity_types=["SPECIES"],
                min_confidence=0.95,
                min_agreement=0.9,
                requires_rule_support=True,
                max_novelty=0.2
            ),
            AutoAcceptRule(
                rule_id="pathogen_high_conf", 
                rule_name="High-confidence pathogen annotations",
                entity_types=["PATHOGEN"],
                min_confidence=0.95,
                min_agreement=0.9,
                requires_rule_support=True,
                max_novelty=0.2
            ),
            AutoAcceptRule(
                rule_id="disease_known",
                rule_name="Known disease entities",
                entity_types=["DISEASE"],
                min_confidence=0.9,
                min_agreement=0.85,
                requires_rule_support=True,
                max_novelty=0.1  # Very conservative for diseases
            ),
            AutoAcceptRule(
                rule_id="measurement_standard",
                rule_name="Standard measurement patterns",
                entity_types=["MEASUREMENT"],
                min_confidence=0.9,
                min_agreement=0.8,
                requires_rule_support=True,
                max_novelty=0.4
            ),
            
            # High-confidence relation rules
            AutoAcceptRule(
                rule_id="causes_relation_strong",
                rule_name="Strong causal relations",
                relation_types=["causes"],
                min_confidence=0.93,
                min_agreement=0.9,
                requires_rule_support=True,
                max_novelty=0.3
            ),
            AutoAcceptRule(
                rule_id="infected_by_clear",
                rule_name="Clear infection relations",
                relation_types=["infected_by"],
                min_confidence=0.9,
                min_agreement=0.85,
                requires_rule_support=True,
                max_novelty=0.3
            ),
            AutoAcceptRule(
                rule_id="measurement_of_standard",
                rule_name="Standard measurement relations",
                relation_types=["measurement_of"],
                min_confidence=0.88,
                min_agreement=0.8,
                requires_rule_support=True,
                max_novelty=0.5
            ),
            
            # Conservative catch-all rule
            AutoAcceptRule(
                rule_id="ultra_high_conf",
                rule_name="Ultra-high confidence any annotation",
                min_confidence=0.98,
                min_agreement=0.95,
                requires_rule_support=True,
                max_novelty=0.1,
                source_authority_min=0.9
            )
        ]
        
        # Load from config file if available
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Parse rules from config
                self.rules = []
                for rule_data in config_data.get("rules", []):
                    rule = AutoAcceptRule(**rule_data)
                    self.rules.append(rule)
                    
                self.global_settings = config_data.get("global_settings", {})
                
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
                self.rules = default_rules
                self.global_settings = {}
        else:
            self.rules = default_rules
            self.global_settings = {
                "enabled": True,
                "max_auto_accept_rate": 0.4,  # Don't auto-accept more than 40% of items
                "min_human_samples": 50,  # Require human samples before enabling
                "review_interval_hours": 24,  # Review performance every 24 hours
                "disable_on_low_precision": 0.85  # Disable rule if precision drops below 85%
            }
    
    def _load_decision_history(self):
        """Load previous auto-accept decisions"""
        if not self.gold_store_path:
            return
            
        history_file = self.gold_store_path.parent / "auto_accept_history.jsonl"
        if history_file.exists():
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        decision_data = json.loads(line)
                        decision = AutoAcceptDecision(**decision_data)
                        self.decisions.append(decision)
                    except Exception as e:
                        logger.warning(f"Failed to load decision: {e}")
    
    def _save_decision(self, decision: AutoAcceptDecision):
        """Save decision to history"""
        if not self.gold_store_path:
            return
            
        history_file = self.gold_store_path.parent / "auto_accept_history.jsonl"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_file, 'a') as f:
            f.write(json.dumps(asdict(decision)) + "\n")
    
    def evaluate_candidate(self,
                          candidate: Dict[str, Any],
                          rule_result: Optional[Dict[str, Any]] = None,
                          triage_item: Optional[Dict[str, Any]] = None) -> AutoAcceptDecision:
        """
        Evaluate a candidate for auto-acceptance.
        
        Args:
            candidate: LLM candidate data
            rule_result: Rule-based result for comparison
            triage_item: Triage item with scores
            
        Returns:
            Auto-accept decision
        """
        item_id = f"{candidate.get('doc_id', 'unknown')}_{candidate.get('sent_id', 'unknown')}"
        
        # Extract candidate properties
        raw_confidence = candidate.get("confidence", 0.0)
        entity_type = candidate.get("label") if "label" in candidate else None
        relation_type = candidate.get("label") if "head_cid" in candidate else None
        
        # Apply confidence calibration
        confidence = self.calibrate_confidence(
            raw_confidence, 
            "llm", 
            entity_type or relation_type
        )
        
        # Calculate agreement with rule-based system
        agreement = self._calculate_agreement(candidate, rule_result)
        
        # Get triage scores
        novelty_score = triage_item.get("novelty_score", 0.5) if triage_item else 0.5
        impact_score = triage_item.get("impact_score", 0.5) if triage_item else 0.5
        authority_score = triage_item.get("authority_score", 0.5) if triage_item else 0.5
        
        # Evaluate against rules
        applicable_rules = []
        decision = "human_review"  # Default to human review
        reasoning_parts = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            # Check if rule applies to this candidate type
            applies = True
            if rule.entity_types and entity_type:
                applies = entity_type in rule.entity_types
            elif rule.relation_types and relation_type:
                applies = relation_type in rule.relation_types
            elif rule.entity_types or rule.relation_types:
                applies = False  # Specific rule that doesn't apply
            
            if not applies:
                continue
            
            # Check all rule conditions
            passes_confidence = confidence >= rule.min_confidence
            passes_agreement = agreement >= rule.min_agreement
            passes_authority = authority_score >= rule.source_authority_min
            passes_novelty = novelty_score <= rule.max_novelty
            passes_rule_support = (not rule.requires_rule_support or 
                                 (rule_result and (rule_result.get("entities") or rule_result.get("relations"))))
            
            # Check rule precision history
            rule_precision_ok = rule.precision >= self.global_settings.get("disable_on_low_precision", 0.85)
            
            conditions = {
                "confidence": passes_confidence,
                "agreement": passes_agreement, 
                "authority": passes_authority,
                "novelty": passes_novelty,
                "rule_support": passes_rule_support,
                "precision_ok": rule_precision_ok
            }
            
            if all(conditions.values()):
                applicable_rules.append(rule.rule_id)
                decision = "auto_accept"
                reasoning_parts.append(f"Rule {rule.rule_name} satisfied")
                break  # First matching rule wins
            else:
                failed_conditions = [k for k, v in conditions.items() if not v]
                reasoning_parts.append(f"Rule {rule.rule_name} failed: {failed_conditions}")
        
        # Global safety checks
        if decision == "auto_accept":
            # Check auto-accept rate
            recent_decisions = [d for d in self.decisions[-100:]]  # Last 100 decisions
            if recent_decisions:
                auto_accept_rate = len([d for d in recent_decisions if d.decision == "auto_accept"]) / len(recent_decisions)
                max_rate = self.global_settings.get("max_auto_accept_rate", 0.4)
                
                if auto_accept_rate > max_rate:
                    decision = "human_review"
                    reasoning_parts.append(f"Auto-accept rate ({auto_accept_rate:.2f}) exceeds limit ({max_rate})")
            
            # Check minimum human samples
            human_samples = len([d for d in recent_decisions if d.decision == "human_review"])
            min_samples = self.global_settings.get("min_human_samples", 50)
            
            if human_samples < min_samples:
                decision = "human_review"
                reasoning_parts.append(f"Need more human samples ({human_samples}/{min_samples})")
        
        # Create decision
        decision_obj = AutoAcceptDecision(
            item_id=item_id,
            decision=decision,
            confidence=confidence,
            applied_rules=applicable_rules,
            reasoning="; ".join(reasoning_parts),
            timestamp=datetime.now().isoformat()
        )
        
        # Save decision
        self.decisions.append(decision_obj)
        self._save_decision(decision_obj)
        
        return decision_obj
    
    def _calculate_agreement(self, 
                           llm_candidate: Dict[str, Any],
                           rule_result: Optional[Dict[str, Any]]) -> float:
        """
        Calculate agreement between LLM and rule-based results.
        
        Args:
            llm_candidate: LLM candidate
            rule_result: Rule-based result
            
        Returns:
            Agreement score (0.0 to 1.0)
        """
        if not rule_result:
            return 0.5  # Neutral when no rule result
        
        # Extract comparable fields
        llm_label = llm_candidate.get("label")
        llm_text = llm_candidate.get("text", "").lower()
        
        # Check entities
        rule_entities = rule_result.get("entities", [])
        for rule_entity in rule_entities:
            rule_label = rule_entity.get("label")
            rule_text = rule_entity.get("text", "").lower()
            
            # Text and label match
            if llm_text == rule_text and llm_label == rule_label:
                return 0.95
            # Label match only
            elif llm_label == rule_label and llm_text in rule_text:
                return 0.8
            # Partial text match
            elif llm_text and rule_text and (llm_text in rule_text or rule_text in llm_text):
                return 0.6
        
        # Check relations
        rule_relations = rule_result.get("relations", [])
        if "head_cid" in llm_candidate and rule_relations:
            llm_rel_label = llm_candidate.get("label")
            for rule_relation in rule_relations:
                rule_rel_label = rule_relation.get("label")
                if llm_rel_label == rule_rel_label:
                    return 0.9
        
        return 0.2  # Low agreement if no matches
    
    def evaluate_batch(self,
                      candidates: List[Dict[str, Any]],
                      rule_results: Optional[List[Dict[str, Any]]] = None,
                      triage_items: Optional[List[Dict[str, Any]]] = None) -> List[AutoAcceptDecision]:
        """Evaluate multiple candidates for auto-acceptance"""
        decisions = []
        
        for i, candidate in enumerate(candidates):
            rule_result = rule_results[i] if rule_results and i < len(rule_results) else None
            triage_item = triage_items[i] if triage_items and i < len(triage_items) else None
            
            decision = self.evaluate_candidate(candidate, rule_result, triage_item)
            decisions.append(decision)
        
        return decisions
    
    def update_rule_performance(self, 
                              rule_id: str,
                              was_correct: bool):
        """
        Update rule performance based on human validation.
        
        Args:
            rule_id: Rule identifier
            was_correct: Whether the auto-accept decision was correct
        """
        # Find rule
        rule = next((r for r in self.rules if r.rule_id == rule_id), None)
        if not rule:
            logger.warning(f"Rule {rule_id} not found")
            return
        
        # Update counters
        if was_correct:
            rule.auto_accepted += 1
        else:
            rule.false_positives += 1
        
        # Recalculate precision
        total = rule.auto_accepted + rule.false_positives
        rule.precision = rule.auto_accepted / total if total > 0 else 1.0
        
        # Disable rule if precision too low
        min_precision = self.global_settings.get("disable_on_low_precision", 0.85)
        if rule.precision < min_precision and total >= 10:  # Need at least 10 samples
            rule.enabled = False
            logger.warning(f"Disabled rule {rule.rule_id} due to low precision: {rule.precision:.3f}")
        
        # Track performance history
        self.performance_history[rule_id].append({
            "timestamp": datetime.now().isoformat(),
            "precision": rule.precision,
            "total_decisions": total,
            "was_correct": was_correct
        })
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get auto-accept system statistics"""
        recent_decisions = self.decisions[-1000:]  # Last 1000 decisions
        
        if not recent_decisions:
            return {"total_decisions": 0}
        
        # Decision distribution
        decision_counts = defaultdict(int)
        for decision in recent_decisions:
            decision_counts[decision.decision] += 1
        
        # Rule usage
        rule_usage = defaultdict(int)
        for decision in recent_decisions:
            for rule_id in decision.applied_rules:
                rule_usage[rule_id] += 1
        
        # Calculate rates
        total = len(recent_decisions)
        auto_accept_rate = decision_counts["auto_accept"] / total
        
        return {
            "total_decisions": len(self.decisions),
            "recent_decisions": total,
            "decision_distribution": dict(decision_counts),
            "auto_accept_rate": auto_accept_rate,
            "rule_usage": dict(rule_usage),
            "active_rules": len([r for r in self.rules if r.enabled]),
            "total_rules": len(self.rules),
            "rule_performance": {
                rule.rule_id: {
                    "precision": rule.precision,
                    "auto_accepted": rule.auto_accepted,
                    "false_positives": rule.false_positives,
                    "enabled": rule.enabled
                }
                for rule in self.rules
            }
        }
    
    def save_config(self, output_path: Optional[Path] = None):
        """Save current configuration to file"""
        output_path = output_path or self.config_path or Path("auto_accept_config.json")
        
        config = {
            "global_settings": self.global_settings,
            "rules": [asdict(rule) for rule in self.rules]
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved auto-accept configuration to {output_path}")
    
    def calibrate_confidence(self, 
                           raw_confidence: float,
                           model_source: str = "llm",
                           entity_type: Optional[str] = None) -> float:
        """
        Calibrate confidence score based on historical performance.
        
        Args:
            raw_confidence: Raw confidence from model
            model_source: Source of confidence ("llm", "rule", "ensemble")
            entity_type: Optional entity type for type-specific calibration
            
        Returns:
            Calibrated confidence score
        """
        # Get calibration history for this model source and entity type
        calibration_key = f"{model_source}_{entity_type or 'all'}"
        
        # Load calibration data if available
        calibration_data = self._get_calibration_data(calibration_key)
        
        if not calibration_data or len(calibration_data) < 10:
            # Not enough data for calibration, return raw confidence with slight penalty
            return max(0.0, raw_confidence - 0.05)
        
        # Apply Platt scaling calibration
        calibrated = self._apply_platt_scaling(raw_confidence, calibration_data)
        
        # Apply reliability adjustment based on variance
        reliability_factor = self._calculate_reliability_factor(calibration_data)
        calibrated = calibrated * reliability_factor
        
        return max(0.0, min(1.0, calibrated))
    
    def _get_calibration_data(self, calibration_key: str) -> List[Dict]:
        """
        Get historical calibration data for a specific model/type combination.
        
        Args:
            calibration_key: Key identifying model source and entity type
            
        Returns:
            List of calibration data points
        """
        if not self.gold_store_path:
            return []
        
        calibration_file = self.gold_store_path.parent / f"calibration_{calibration_key}.json"
        if not calibration_file.exists():
            return []
        
        try:
            with open(calibration_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}")
            return []
    
    def _apply_platt_scaling(self, 
                           raw_confidence: float, 
                           calibration_data: List[Dict]) -> float:
        """
        Apply Platt scaling to calibrate confidence scores.
        
        Args:
            raw_confidence: Raw confidence score
            calibration_data: Historical accuracy data
            
        Returns:
            Calibrated confidence score
        """
        # Group data into confidence bins
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_accuracies = []
        
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            bin_data = [
                point for point in calibration_data
                if bin_start <= point['confidence'] < bin_end
            ]
            
            if bin_data:
                accuracy = sum(point['was_correct'] for point in bin_data) / len(bin_data)
                bin_accuracies.append(accuracy)
            else:
                # No data for this bin, use raw confidence as estimate
                bin_accuracies.append((bin_start + bin_end) / 2)
        
        # Find the appropriate bin for the raw confidence
        bin_index = min(int(raw_confidence * 10), len(bin_accuracies) - 1)
        
        # Interpolate between bins if possible
        if bin_index < len(bin_accuracies) - 1:
            bin_low = bin_accuracies[bin_index]
            bin_high = bin_accuracies[bin_index + 1]
            
            # Linear interpolation within the bin
            bin_position = (raw_confidence * 10) - bin_index
            calibrated = bin_low + (bin_high - bin_low) * bin_position
        else:
            calibrated = bin_accuracies[bin_index]
        
        return calibrated
    
    def _calculate_reliability_factor(self, calibration_data: List[Dict]) -> float:
        """
        Calculate reliability factor based on calibration data variance.
        
        Args:
            calibration_data: Historical calibration data
            
        Returns:
            Reliability factor (0.5 to 1.0)
        """
        if len(calibration_data) < 5:
            return 0.8  # Conservative factor for small samples
        
        # Calculate accuracy variance across recent samples
        recent_data = calibration_data[-50:]  # Last 50 decisions
        accuracies = [point['was_correct'] for point in recent_data]
        
        if not accuracies:
            return 0.8
        
        mean_accuracy = statistics.mean(accuracies)
        
        # Calculate variance (lower variance = higher reliability)
        if len(accuracies) > 1:
            variance = statistics.variance(accuracies)
            # Convert variance to reliability factor (inverse relationship)
            reliability = max(0.5, 1.0 - variance)
        else:
            reliability = 0.8
        
        return reliability
    
    def update_calibration_data(self,
                              confidence: float,
                              was_correct: bool,
                              model_source: str = "llm",
                              entity_type: Optional[str] = None):
        """
        Update calibration data with new validation result.
        
        Args:
            confidence: Confidence score that was used
            was_correct: Whether the prediction was correct
            model_source: Source of the confidence score
            entity_type: Optional entity type
        """
        calibration_key = f"{model_source}_{entity_type or 'all'}"
        
        # Load existing data
        calibration_data = self._get_calibration_data(calibration_key)
        
        # Add new data point
        new_point = {
            "confidence": confidence,
            "was_correct": was_correct,
            "timestamp": datetime.now().isoformat(),
            "model_source": model_source,
            "entity_type": entity_type
        }
        
        calibration_data.append(new_point)
        
        # Keep only recent data (last 1000 points)
        if len(calibration_data) > 1000:
            calibration_data = calibration_data[-1000:]
        
        # Save updated data
        if self.gold_store_path:
            calibration_file = self.gold_store_path.parent / f"calibration_{calibration_key}.json"
            calibration_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(calibration_file, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save calibration data: {e}")
    
    def get_calibration_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about confidence calibration across all model sources.
        
        Returns:
            Dictionary with calibration statistics
        """
        if not self.gold_store_path:
            return {}
        
        stats = {}
        calibration_dir = self.gold_store_path.parent
        
        # Find all calibration files
        calibration_files = list(calibration_dir.glob("calibration_*.json"))
        
        for cal_file in calibration_files:
            # Extract calibration key from filename
            cal_key = cal_file.stem.replace("calibration_", "")
            
            # Load calibration data
            cal_data = self._get_calibration_data(cal_key)
            
            if cal_data:
                # Calculate statistics
                confidences = [point['confidence'] for point in cal_data]
                accuracies = [point['was_correct'] for point in cal_data]
                
                # Calibration error (ECE - Expected Calibration Error)
                ece = self._calculate_expected_calibration_error(cal_data)
                
                stats[cal_key] = {
                    "total_samples": len(cal_data),
                    "mean_confidence": statistics.mean(confidences),
                    "mean_accuracy": statistics.mean(accuracies),
                    "expected_calibration_error": ece,
                    "confidence_variance": statistics.variance(confidences) if len(confidences) > 1 else 0,
                    "accuracy_variance": statistics.variance(accuracies) if len(accuracies) > 1 else 0
                }
        
        return stats
    
    def _calculate_expected_calibration_error(self, calibration_data: List[Dict]) -> float:
        """
        Calculate Expected Calibration Error (ECE) for the calibration data.
        
        Args:
            calibration_data: Historical calibration data
            
        Returns:
            Expected Calibration Error (lower is better)
        """
        if not calibration_data:
            return 1.0
        
        # Create confidence bins
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        total_samples = len(calibration_data)
        weighted_error = 0.0
        
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            bin_data = [
                point for point in calibration_data
                if bin_start <= point['confidence'] < bin_end
            ]
            
            if bin_data:
                bin_confidence = statistics.mean([point['confidence'] for point in bin_data])
                bin_accuracy = statistics.mean([point['was_correct'] for point in bin_data])
                bin_weight = len(bin_data) / total_samples
                
                # Contribution to ECE
                weighted_error += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return weighted_error


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-accept system")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--test", action="store_true", help="Run test evaluation")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--update-rule", help="Update rule performance (rule_id:correct)")
    
    args = parser.parse_args()
    
    # Initialize system
    config_path = Path(args.config) if args.config else None
    system = AutoAcceptSystem(config_path)
    
    if args.test:
        # Test evaluation
        test_candidate = {
            "text": "Vibrio parahaemolyticus",
            "label": "PATHOGEN",
            "confidence": 0.96,
            "doc_id": "test_doc",
            "sent_id": "s1"
        }
        
        test_rule_result = {
            "entities": [
                {"text": "Vibrio parahaemolyticus", "label": "PATHOGEN", "confidence": 0.9}
            ]
        }
        
        decision = system.evaluate_candidate(test_candidate, test_rule_result)
        print(f"Decision: {decision.decision}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Applied rules: {decision.applied_rules}")
    
    if args.stats:
        # Show statistics
        stats = system.get_system_statistics()
        print(json.dumps(stats, indent=2))
    
    if args.update_rule:
        # Update rule performance
        parts = args.update_rule.split(":")
        if len(parts) == 2:
            rule_id, correct = parts
            system.update_rule_performance(rule_id, correct.lower() == "true")
            print(f"Updated rule {rule_id} performance")
        else:
            print("Format: rule_id:true/false")
"""
Automated Model Retraining Trigger System

Monitors annotation quality, data volume, and performance metrics to automatically
trigger model retraining when conditions are met. Integrates with the event system
to provide intelligent, data-driven retraining decisions.
"""

import json
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..database.models import GoldAnnotation, ModelTrainingRun, Document, TriageItem
from ..database.connection import get_db_session
from ..events.event_system import EventPublisher, EventType

logger = logging.getLogger(__name__)

class TriggerCondition(Enum):
    """Model retraining trigger conditions"""
    DATA_VOLUME_THRESHOLD = "data_volume_threshold"
    QUALITY_DEGRADATION = "quality_degradation"
    TIME_BASED_SCHEDULE = "time_based_schedule"
    PERFORMANCE_DROP = "performance_drop"
    NEW_DOMAIN_DATA = "new_domain_data"
    ANNOTATION_DISAGREEMENT = "annotation_disagreement"
    MANUAL_TRIGGER = "manual_trigger"

class ModelType(Enum):
    """Types of models that can be retrained"""
    SCIBERT_NER = "scibert_ner"
    RELATION_EXTRACTION = "relation_extraction"
    TOPIC_CLASSIFICATION = "topic_classification"
    AUTO_ACCEPT_CLASSIFIER = "auto_accept_classifier"

@dataclass
class TriggerConfig:
    """Configuration for retraining triggers"""
    # Data volume thresholds
    min_new_annotations: int = 100
    min_annotation_ratio: float = 0.1  # 10% new data since last training
    
    # Quality thresholds
    min_quality_drop: float = 0.05  # 5% drop in F1 score
    min_iaa_drop: float = 0.1  # 10% drop in Inter-Annotator Agreement
    
    # Time-based scheduling
    max_days_since_training: int = 30
    min_days_between_training: int = 7
    
    # Performance thresholds
    min_performance_drop: float = 0.05  # 5% drop in auto-accept accuracy
    
    # Domain detection
    novelty_threshold: float = 0.3  # 30% novel entities/relations
    
    # System settings
    enable_automatic_training: bool = True
    require_manual_approval: bool = False
    parallel_training_limit: int = 2

@dataclass
class RetrainingDecision:
    """Decision to retrain a model"""
    decision_id: str
    model_type: ModelType
    trigger_conditions: List[TriggerCondition]
    confidence_score: float
    data_snapshot: Dict[str, Any]
    estimated_duration: int  # minutes
    priority: int  # 1-10, 10 being highest
    created_at: datetime
    approved_by: Optional[str] = None
    executed_at: Optional[datetime] = None

class ModelRetrainingOrchestrator:
    """Main orchestrator for automated model retraining"""
    
    def __init__(self, 
                 config: TriggerConfig = None,
                 redis_url: str = "redis://localhost:6379"):
        """
        Initialize retraining orchestrator.
        
        Args:
            config: Trigger configuration
            redis_url: Redis URL for event publishing
        """
        self.config = config or TriggerConfig()
        self.event_publisher = EventPublisher(redis_url)
        self.active_decisions: Dict[str, RetrainingDecision] = {}
        self.running = False
        
    async def start_monitoring(self):
        """Start the monitoring loop"""
        await self.event_publisher.connect()
        self.running = True
        
        logger.info("Started model retraining orchestrator")
        
        while self.running:
            try:
                # Check all trigger conditions
                await self._check_all_triggers()
                
                # Process pending decisions
                await self._process_decisions()
                
                # Clean up completed decisions
                self._cleanup_decisions()
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in retraining orchestrator: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.running = False
    
    async def _check_all_triggers(self):
        """Check all configured trigger conditions"""
        for model_type in ModelType:
            try:
                decision = await self._evaluate_model_retraining(model_type)
                if decision and decision.confidence_score >= 0.7:
                    await self._handle_retraining_decision(decision)
            except Exception as e:
                logger.error(f"Error checking triggers for {model_type.value}: {e}")
    
    async def _evaluate_model_retraining(self, model_type: ModelType) -> Optional[RetrainingDecision]:
        """
        Evaluate if a model should be retrained.
        
        Args:
            model_type: Type of model to evaluate
            
        Returns:
            RetrainingDecision if retraining is recommended
        """
        triggered_conditions = []
        confidence_factors = []
        data_snapshot = {}
        
        with get_db_session() as db:
            # Get last training info
            last_training = db.query(ModelTrainingRun)\
                             .filter(ModelTrainingRun.model_type == model_type.value)\
                             .order_by(ModelTrainingRun.created_at.desc())\
                             .first()
            
            last_training_date = last_training.created_at if last_training else None
            
            # Check data volume trigger
            volume_trigger, volume_confidence, volume_data = await self._check_data_volume_trigger(
                db, model_type, last_training_date
            )
            if volume_trigger:
                triggered_conditions.append(TriggerCondition.DATA_VOLUME_THRESHOLD)
                confidence_factors.append(volume_confidence)
                data_snapshot.update(volume_data)
            
            # Check quality degradation trigger
            quality_trigger, quality_confidence, quality_data = await self._check_quality_trigger(
                db, model_type, last_training_date
            )
            if quality_trigger:
                triggered_conditions.append(TriggerCondition.QUALITY_DEGRADATION)
                confidence_factors.append(quality_confidence)
                data_snapshot.update(quality_data)
            
            # Check time-based trigger
            time_trigger, time_confidence, time_data = await self._check_time_trigger(
                db, model_type, last_training_date
            )
            if time_trigger:
                triggered_conditions.append(TriggerCondition.TIME_BASED_SCHEDULE)
                confidence_factors.append(time_confidence)
                data_snapshot.update(time_data)
            
            # Check performance drop trigger
            perf_trigger, perf_confidence, perf_data = await self._check_performance_trigger(
                db, model_type, last_training_date
            )
            if perf_trigger:
                triggered_conditions.append(TriggerCondition.PERFORMANCE_DROP)
                confidence_factors.append(perf_confidence)
                data_snapshot.update(perf_data)
            
            # Check new domain data trigger
            domain_trigger, domain_confidence, domain_data = await self._check_domain_trigger(
                db, model_type, last_training_date
            )
            if domain_trigger:
                triggered_conditions.append(TriggerCondition.NEW_DOMAIN_DATA)
                confidence_factors.append(domain_confidence)
                data_snapshot.update(domain_data)
        
        # Calculate overall confidence and priority
        if not triggered_conditions:
            return None
        
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        priority = self._calculate_priority(triggered_conditions, overall_confidence)
        estimated_duration = self._estimate_training_duration(model_type, data_snapshot)
        
        decision = RetrainingDecision(
            decision_id=str(uuid.uuid4()),
            model_type=model_type,
            trigger_conditions=triggered_conditions,
            confidence_score=overall_confidence,
            data_snapshot=data_snapshot,
            estimated_duration=estimated_duration,
            priority=priority,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Retraining decision for {model_type.value}: "
                   f"confidence={overall_confidence:.2f}, conditions={[c.value for c in triggered_conditions]}")
        
        return decision
    
    async def _check_data_volume_trigger(self, 
                                       db: Session, 
                                       model_type: ModelType, 
                                       last_training_date: Optional[datetime]) -> Tuple[bool, float, Dict]:
        """Check if data volume threshold is met"""
        data_snapshot = {}
        
        # Count total annotations
        total_annotations = db.query(func.count(GoldAnnotation.id)).scalar()
        data_snapshot["total_annotations"] = total_annotations
        
        if last_training_date:
            # Count new annotations since last training
            new_annotations = db.query(func.count(GoldAnnotation.id))\
                               .filter(GoldAnnotation.created_at > last_training_date)\
                               .scalar()
            
            data_snapshot["new_annotations"] = new_annotations
            data_snapshot["annotation_ratio"] = new_annotations / max(total_annotations, 1)
            
            # Check absolute and ratio thresholds
            volume_trigger = (
                new_annotations >= self.config.min_new_annotations and
                data_snapshot["annotation_ratio"] >= self.config.min_annotation_ratio
            )
            
            # Calculate confidence based on how much we exceed thresholds
            abs_confidence = min(new_annotations / self.config.min_new_annotations, 2.0) / 2.0
            ratio_confidence = min(data_snapshot["annotation_ratio"] / self.config.min_annotation_ratio, 2.0) / 2.0
            confidence = (abs_confidence + ratio_confidence) / 2.0
            
        else:
            # First training - trigger if we have minimum data
            volume_trigger = total_annotations >= self.config.min_new_annotations
            confidence = min(total_annotations / self.config.min_new_annotations, 1.0)
            data_snapshot["new_annotations"] = total_annotations
            data_snapshot["annotation_ratio"] = 1.0
        
        return volume_trigger, confidence, data_snapshot
    
    async def _check_quality_trigger(self, 
                                   db: Session, 
                                   model_type: ModelType, 
                                   last_training_date: Optional[datetime]) -> Tuple[bool, float, Dict]:
        """Check if quality has degraded significantly"""
        data_snapshot = {}
        
        if not last_training_date:
            return False, 0.0, data_snapshot
        
        # Calculate recent vs historical Inter-Annotator Agreement
        # This would require implementing IAA calculation logic
        # For now, we'll use a placeholder
        recent_iaa = 0.85  # Placeholder
        historical_iaa = 0.92  # Placeholder
        
        iaa_drop = historical_iaa - recent_iaa
        data_snapshot["recent_iaa"] = recent_iaa
        data_snapshot["historical_iaa"] = historical_iaa
        data_snapshot["iaa_drop"] = iaa_drop
        
        quality_trigger = iaa_drop >= self.config.min_iaa_drop
        confidence = min(iaa_drop / self.config.min_iaa_drop, 1.0) if quality_trigger else 0.0
        
        return quality_trigger, confidence, data_snapshot
    
    async def _check_time_trigger(self, 
                                db: Session, 
                                model_type: ModelType, 
                                last_training_date: Optional[datetime]) -> Tuple[bool, float, Dict]:
        """Check if enough time has passed since last training"""
        data_snapshot = {}
        
        if not last_training_date:
            return True, 0.8, {"reason": "no_previous_training"}
        
        days_since_training = (datetime.utcnow() - last_training_date).days
        data_snapshot["days_since_training"] = days_since_training
        data_snapshot["max_days_configured"] = self.config.max_days_since_training
        
        # Check if we're past the maximum allowed time
        time_trigger = days_since_training >= self.config.max_days_since_training
        
        # Check if minimum time between trainings has passed
        min_time_passed = days_since_training >= self.config.min_days_between_training
        
        if time_trigger and min_time_passed:
            confidence = min(days_since_training / self.config.max_days_since_training, 1.0)
            return True, confidence, data_snapshot
        
        return False, 0.0, data_snapshot
    
    async def _check_performance_trigger(self, 
                                       db: Session, 
                                       model_type: ModelType, 
                                       last_training_date: Optional[datetime]) -> Tuple[bool, float, Dict]:
        """Check if model performance has degraded"""
        data_snapshot = {}
        
        if model_type != ModelType.AUTO_ACCEPT_CLASSIFIER:
            return False, 0.0, data_snapshot
        
        # Calculate recent auto-accept accuracy
        # This would require implementing performance tracking
        # For now, we'll use a placeholder
        recent_accuracy = 0.88  # Placeholder
        baseline_accuracy = 0.94  # Placeholder
        
        performance_drop = baseline_accuracy - recent_accuracy
        data_snapshot["recent_accuracy"] = recent_accuracy
        data_snapshot["baseline_accuracy"] = baseline_accuracy
        data_snapshot["performance_drop"] = performance_drop
        
        perf_trigger = performance_drop >= self.config.min_performance_drop
        confidence = min(performance_drop / self.config.min_performance_drop, 1.0) if perf_trigger else 0.0
        
        return perf_trigger, confidence, data_snapshot
    
    async def _check_domain_trigger(self, 
                                  db: Session, 
                                  model_type: ModelType, 
                                  last_training_date: Optional[datetime]) -> Tuple[bool, float, Dict]:
        """Check if significant new domain data has been added"""
        data_snapshot = {}
        
        if not last_training_date:
            return False, 0.0, data_snapshot
        
        # Analyze novelty of recent annotations
        # This would require implementing semantic analysis
        # For now, we'll use a placeholder based on new entity types
        
        # Count unique entity types in recent vs historical data
        recent_entities = db.query(GoldAnnotation.entity_type)\
                           .filter(GoldAnnotation.created_at > last_training_date)\
                           .distinct().count()
        
        historical_entities = db.query(GoldAnnotation.entity_type)\
                             .filter(GoldAnnotation.created_at <= last_training_date)\
                             .distinct().count()
        
        novelty_ratio = recent_entities / max(historical_entities, 1)
        data_snapshot["recent_entity_types"] = recent_entities
        data_snapshot["historical_entity_types"] = historical_entities
        data_snapshot["novelty_ratio"] = novelty_ratio
        
        domain_trigger = novelty_ratio >= self.config.novelty_threshold
        confidence = min(novelty_ratio / self.config.novelty_threshold, 1.0) if domain_trigger else 0.0
        
        return domain_trigger, confidence, data_snapshot
    
    def _calculate_priority(self, 
                           triggered_conditions: List[TriggerCondition], 
                           confidence: float) -> int:
        """Calculate priority score (1-10) for retraining decision"""
        base_priority = int(confidence * 5)  # 0-5 from confidence
        
        # Adjust based on trigger types
        condition_weights = {
            TriggerCondition.PERFORMANCE_DROP: 3,
            TriggerCondition.QUALITY_DEGRADATION: 2,
            TriggerCondition.NEW_DOMAIN_DATA: 2,
            TriggerCondition.DATA_VOLUME_THRESHOLD: 1,
            TriggerCondition.TIME_BASED_SCHEDULE: 1,
        }
        
        condition_bonus = sum(condition_weights.get(c, 0) for c in triggered_conditions)
        
        return min(base_priority + condition_bonus, 10)
    
    def _estimate_training_duration(self, 
                                   model_type: ModelType, 
                                   data_snapshot: Dict[str, Any]) -> int:
        """Estimate training duration in minutes"""
        base_durations = {
            ModelType.SCIBERT_NER: 120,  # 2 hours
            ModelType.RELATION_EXTRACTION: 90,  # 1.5 hours
            ModelType.TOPIC_CLASSIFICATION: 60,  # 1 hour
            ModelType.AUTO_ACCEPT_CLASSIFIER: 30,  # 30 minutes
        }
        
        base_duration = base_durations.get(model_type, 60)
        
        # Scale by data volume
        total_annotations = data_snapshot.get("total_annotations", 1000)
        scale_factor = min(total_annotations / 1000, 3.0)  # Cap at 3x
        
        return int(base_duration * scale_factor)
    
    async def _handle_retraining_decision(self, decision: RetrainingDecision):
        """Handle a retraining decision"""
        if not self.config.enable_automatic_training:
            logger.info(f"Automatic training disabled, skipping {decision.model_type.value}")
            return
        
        if self.config.require_manual_approval:
            # Store decision for manual approval
            self.active_decisions[decision.decision_id] = decision
            
            # Publish event for manual review
            await self.event_publisher.publish_simple(
                EventType.MODEL_TRAINING_TRIGGERED,
                {
                    "decision_id": decision.decision_id,
                    "model_type": decision.model_type.value,
                    "requires_approval": True,
                    "confidence": decision.confidence_score,
                    "conditions": [c.value for c in decision.trigger_conditions]
                },
                "retraining_orchestrator"
            )
            
            logger.info(f"Retraining decision {decision.decision_id} requires manual approval")
        else:
            # Execute immediately
            await self._execute_retraining(decision)
    
    async def _execute_retraining(self, decision: RetrainingDecision):
        """Execute a retraining decision"""
        try:
            # Check parallel training limit
            active_trainings = len([d for d in self.active_decisions.values() 
                                  if d.executed_at and not self._is_training_complete(d)])
            
            if active_trainings >= self.config.parallel_training_limit:
                logger.info(f"Parallel training limit reached, queuing {decision.decision_id}")
                self.active_decisions[decision.decision_id] = decision
                return
            
            decision.executed_at = datetime.utcnow()
            self.active_decisions[decision.decision_id] = decision
            
            # Create training run record
            with get_db_session() as db:
                training_run = ModelTrainingRun(
                    id=str(uuid.uuid4()),
                    model_type=decision.model_type.value,
                    trigger_conditions=json.dumps([c.value for c in decision.trigger_conditions]),
                    confidence_score=decision.confidence_score,
                    data_snapshot=json.dumps(decision.data_snapshot),
                    status="initiated",
                    created_at=decision.executed_at
                )
                db.add(training_run)
                db.commit()
            
            # Publish training event
            await self.event_publisher.publish_simple(
                EventType.MODEL_TRAINING_TRIGGERED,
                {
                    "decision_id": decision.decision_id,
                    "training_run_id": training_run.id,
                    "model_type": decision.model_type.value,
                    "estimated_duration": decision.estimated_duration,
                    "data_snapshot": decision.data_snapshot
                },
                "retraining_orchestrator"
            )
            
            logger.info(f"Started retraining for {decision.model_type.value} (decision: {decision.decision_id})")
            
        except Exception as e:
            logger.error(f"Failed to execute retraining decision {decision.decision_id}: {e}")
    
    async def approve_retraining(self, decision_id: str, approved_by: str) -> bool:
        """Manually approve a retraining decision"""
        if decision_id not in self.active_decisions:
            return False
        
        decision = self.active_decisions[decision_id]
        decision.approved_by = approved_by
        
        await self._execute_retraining(decision)
        return True
    
    async def reject_retraining(self, decision_id: str, rejected_by: str) -> bool:
        """Manually reject a retraining decision"""
        if decision_id not in self.active_decisions:
            return False
        
        decision = self.active_decisions[decision_id]
        
        # Log rejection
        logger.info(f"Retraining decision {decision_id} rejected by {rejected_by}")
        
        # Remove from active decisions
        del self.active_decisions[decision_id]
        return True
    
    async def _process_decisions(self):
        """Process pending retraining decisions"""
        for decision_id, decision in list(self.active_decisions.items()):
            if not decision.executed_at:
                # Check if we can execute now
                active_trainings = len([d for d in self.active_decisions.values() 
                                      if d.executed_at and not self._is_training_complete(d)])
                
                if active_trainings < self.config.parallel_training_limit:
                    if not self.config.require_manual_approval or decision.approved_by:
                        await self._execute_retraining(decision)
    
    def _is_training_complete(self, decision: RetrainingDecision) -> bool:
        """Check if training is complete (placeholder)"""
        if not decision.executed_at:
            return False
        
        # This would check actual training status
        # For now, assume training completes after estimated duration
        elapsed = (datetime.utcnow() - decision.executed_at).total_seconds() / 60
        return elapsed >= decision.estimated_duration
    
    def _cleanup_decisions(self):
        """Clean up completed decisions"""
        completed = [
            decision_id for decision_id, decision in self.active_decisions.items()
            if self._is_training_complete(decision)
        ]
        
        for decision_id in completed:
            del self.active_decisions[decision_id]
    
    def get_active_decisions(self) -> List[Dict[str, Any]]:
        """Get all active retraining decisions"""
        return [asdict(decision) for decision in self.active_decisions.values()]
    
    def get_decision_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific decision"""
        if decision_id not in self.active_decisions:
            return None
        
        decision = self.active_decisions[decision_id]
        status = asdict(decision)
        status["is_complete"] = self._is_training_complete(decision)
        
        return status

class RetrainingTriggerAPI:
    """API interface for retraining triggers"""
    
    def __init__(self, orchestrator: ModelRetrainingOrchestrator):
        self.orchestrator = orchestrator
    
    async def manual_trigger_retraining(self, 
                                      model_type: ModelType, 
                                      triggered_by: str,
                                      reason: str = "Manual trigger") -> str:
        """Manually trigger model retraining"""
        decision = RetrainingDecision(
            decision_id=str(uuid.uuid4()),
            model_type=model_type,
            trigger_conditions=[TriggerCondition.MANUAL_TRIGGER],
            confidence_score=1.0,
            data_snapshot={"manual_reason": reason, "triggered_by": triggered_by},
            estimated_duration=self.orchestrator._estimate_training_duration(model_type, {}),
            priority=8,  # High priority for manual triggers
            created_at=datetime.utcnow(),
            approved_by=triggered_by
        )
        
        await self.orchestrator._handle_retraining_decision(decision)
        return decision.decision_id
    
    async def get_retraining_history(self, 
                                   model_type: Optional[ModelType] = None,
                                   limit: int = 50) -> List[Dict[str, Any]]:
        """Get retraining history"""
        with get_db_session() as db:
            query = db.query(ModelTrainingRun).order_by(ModelTrainingRun.created_at.desc())
            
            if model_type:
                query = query.filter(ModelTrainingRun.model_type == model_type.value)
            
            runs = query.limit(limit).all()
            
            return [
                {
                    "id": run.id,
                    "model_type": run.model_type,
                    "trigger_conditions": json.loads(run.trigger_conditions or "[]"),
                    "confidence_score": run.confidence_score,
                    "status": run.status,
                    "created_at": run.created_at.isoformat(),
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None
                }
                for run in runs
            ]
    
    async def update_trigger_config(self, new_config: Dict[str, Any]) -> bool:
        """Update trigger configuration"""
        try:
            # Validate and update config
            for key, value in new_config.items():
                if hasattr(self.orchestrator.config, key):
                    setattr(self.orchestrator.config, key, value)
            
            logger.info(f"Updated trigger configuration: {new_config}")
            return True
        except Exception as e:
            logger.error(f"Failed to update trigger config: {e}")
            return False

# Factory function
def create_retraining_orchestrator(config: Optional[TriggerConfig] = None,
                                 redis_url: str = "redis://localhost:6379") -> ModelRetrainingOrchestrator:
    """Create a model retraining orchestrator"""
    return ModelRetrainingOrchestrator(config, redis_url)

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model retraining trigger system")
    parser.add_argument("--action", choices=["monitor", "trigger", "status", "history"], required=True)
    parser.add_argument("--model-type", choices=[e.value for e in ModelType])
    parser.add_argument("--decision-id")
    parser.add_argument("--config-file", help="Path to configuration file")
    
    args = parser.parse_args()
    
    async def main():
        # Load config
        config = TriggerConfig()
        if args.config_file:
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        orchestrator = create_retraining_orchestrator(config)
        api = RetrainingTriggerAPI(orchestrator)
        
        if args.action == "monitor":
            print("Starting retraining monitor...")
            await orchestrator.start_monitoring()
        
        elif args.action == "trigger":
            if not args.model_type:
                print("--model-type required for trigger action")
                return
            
            model_type = ModelType(args.model_type)
            decision_id = await api.manual_trigger_retraining(
                model_type, "cli_user", "Manual CLI trigger"
            )
            print(f"Triggered retraining: {decision_id}")
        
        elif args.action == "status":
            if args.decision_id:
                status = orchestrator.get_decision_status(args.decision_id)
                print(json.dumps(status, indent=2, default=str))
            else:
                decisions = orchestrator.get_active_decisions()
                print(json.dumps(decisions, indent=2, default=str))
        
        elif args.action == "history":
            model_type = ModelType(args.model_type) if args.model_type else None
            history = await api.get_retraining_history(model_type)
            print(json.dumps(history, indent=2))
    
    asyncio.run(main())
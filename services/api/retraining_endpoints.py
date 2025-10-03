"""
REST API Endpoints for Model Retraining Triggers

Provides HTTP endpoints for managing automated model retraining,
including manual triggers, status monitoring, and configuration management.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database.connection import get_db_session
from ..training.model_retraining_triggers import (
    ModelRetrainingOrchestrator,
    RetrainingTriggerAPI,
    TriggerConfig,
    ModelType,
    TriggerCondition,
    create_retraining_orchestrator
)

logger = logging.getLogger(__name__)

# Pydantic models for API
class TriggerConfigRequest(BaseModel):
    """Request model for updating trigger configuration"""
    min_new_annotations: Optional[int] = Field(None, ge=10, le=1000)
    min_annotation_ratio: Optional[float] = Field(None, ge=0.01, le=1.0)
    min_quality_drop: Optional[float] = Field(None, ge=0.01, le=0.5)
    min_iaa_drop: Optional[float] = Field(None, ge=0.01, le=0.5)
    max_days_since_training: Optional[int] = Field(None, ge=1, le=365)
    min_days_between_training: Optional[int] = Field(None, ge=1, le=30)
    min_performance_drop: Optional[float] = Field(None, ge=0.01, le=0.5)
    novelty_threshold: Optional[float] = Field(None, ge=0.1, le=1.0)
    enable_automatic_training: Optional[bool] = None
    require_manual_approval: Optional[bool] = None
    parallel_training_limit: Optional[int] = Field(None, ge=1, le=5)

class ManualTriggerRequest(BaseModel):
    """Request model for manual retraining trigger"""
    model_type: str = Field(..., description="Type of model to retrain")
    reason: str = Field(..., min_length=10, max_length=500, description="Reason for manual trigger")
    triggered_by: str = Field(..., description="User email or identifier")

class RetrainingDecisionResponse(BaseModel):
    """Response model for retraining decisions"""
    decision_id: str
    model_type: str
    trigger_conditions: List[str]
    confidence_score: float
    priority: int
    estimated_duration: int
    created_at: str
    executed_at: Optional[str] = None
    approved_by: Optional[str] = None
    is_complete: Optional[bool] = None

class RetrainingHistoryResponse(BaseModel):
    """Response model for retraining history"""
    id: str
    model_type: str
    trigger_conditions: List[str]
    confidence_score: Optional[float]
    status: str
    created_at: str
    completed_at: Optional[str] = None

class RetrainingStatusResponse(BaseModel):
    """Response model for overall retraining status"""
    orchestrator_running: bool
    active_decisions: List[RetrainingDecisionResponse]
    config: Dict[str, Any]
    recent_runs: List[RetrainingHistoryResponse]

# Global orchestrator instance
_orchestrator: Optional[ModelRetrainingOrchestrator] = None
_trigger_api: Optional[RetrainingTriggerAPI] = None

def get_orchestrator() -> ModelRetrainingOrchestrator:
    """Get or create the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_retraining_orchestrator()
    return _orchestrator

def get_trigger_api() -> RetrainingTriggerAPI:
    """Get or create the global trigger API instance"""
    global _trigger_api
    if _trigger_api is None:
        _trigger_api = RetrainingTriggerAPI(get_orchestrator())
    return _trigger_api

# Router setup
router = APIRouter(prefix="/retraining", tags=["Model Retraining"])

@router.get("/status", response_model=RetrainingStatusResponse)
async def get_retraining_status():
    """Get overall retraining system status"""
    try:
        orchestrator = get_orchestrator()
        trigger_api = get_trigger_api()
        
        # Get active decisions
        active_decisions_raw = orchestrator.get_active_decisions()
        active_decisions = [
            RetrainingDecisionResponse(
                decision_id=d["decision_id"],
                model_type=d["model_type"].value if hasattr(d["model_type"], "value") else d["model_type"],
                trigger_conditions=[c.value if hasattr(c, "value") else c for c in d["trigger_conditions"]],
                confidence_score=d["confidence_score"],
                priority=d["priority"],
                estimated_duration=d["estimated_duration"],
                created_at=d["created_at"].isoformat() if isinstance(d["created_at"], datetime) else d["created_at"],
                executed_at=d["executed_at"].isoformat() if d.get("executed_at") and isinstance(d["executed_at"], datetime) else d.get("executed_at"),
                approved_by=d.get("approved_by")
            )
            for d in active_decisions_raw
        ]
        
        # Get recent runs
        recent_runs_raw = await trigger_api.get_retraining_history(limit=10)
        recent_runs = [
            RetrainingHistoryResponse(**run) for run in recent_runs_raw
        ]
        
        # Get current config
        config = {
            "min_new_annotations": orchestrator.config.min_new_annotations,
            "min_annotation_ratio": orchestrator.config.min_annotation_ratio,
            "min_quality_drop": orchestrator.config.min_quality_drop,
            "min_iaa_drop": orchestrator.config.min_iaa_drop,
            "max_days_since_training": orchestrator.config.max_days_since_training,
            "min_days_between_training": orchestrator.config.min_days_between_training,
            "min_performance_drop": orchestrator.config.min_performance_drop,
            "novelty_threshold": orchestrator.config.novelty_threshold,
            "enable_automatic_training": orchestrator.config.enable_automatic_training,
            "require_manual_approval": orchestrator.config.require_manual_approval,
            "parallel_training_limit": orchestrator.config.parallel_training_limit,
        }
        
        return RetrainingStatusResponse(
            orchestrator_running=orchestrator.running,
            active_decisions=active_decisions,
            config=config,
            recent_runs=recent_runs
        )
        
    except Exception as e:
        logger.error(f"Error getting retraining status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trigger", response_model=Dict[str, str])
async def manual_trigger_retraining(request: ManualTriggerRequest):
    """Manually trigger model retraining"""
    try:
        # Validate model type
        try:
            model_type = ModelType(request.model_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model_type. Must be one of: {[e.value for e in ModelType]}"
            )
        
        trigger_api = get_trigger_api()
        
        decision_id = await trigger_api.manual_trigger_retraining(
            model_type=model_type,
            triggered_by=request.triggered_by,
            reason=request.reason
        )
        
        logger.info(f"Manual retraining triggered by {request.triggered_by} for {request.model_type}: {decision_id}")
        
        return {"decision_id": decision_id, "message": "Retraining triggered successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering manual retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decisions/{decision_id}", response_model=RetrainingDecisionResponse)
async def get_decision_status(decision_id: str):
    """Get status of a specific retraining decision"""
    try:
        orchestrator = get_orchestrator()
        decision_data = orchestrator.get_decision_status(decision_id)
        
        if not decision_data:
            raise HTTPException(status_code=404, detail="Decision not found")
        
        return RetrainingDecisionResponse(
            decision_id=decision_data["decision_id"],
            model_type=decision_data["model_type"].value if hasattr(decision_data["model_type"], "value") else decision_data["model_type"],
            trigger_conditions=[c.value if hasattr(c, "value") else c for c in decision_data["trigger_conditions"]],
            confidence_score=decision_data["confidence_score"],
            priority=decision_data["priority"],
            estimated_duration=decision_data["estimated_duration"],
            created_at=decision_data["created_at"].isoformat() if isinstance(decision_data["created_at"], datetime) else decision_data["created_at"],
            executed_at=decision_data["executed_at"].isoformat() if decision_data.get("executed_at") and isinstance(decision_data["executed_at"], datetime) else decision_data.get("executed_at"),
            approved_by=decision_data.get("approved_by"),
            is_complete=decision_data.get("is_complete", False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decision status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decisions/{decision_id}/approve")
async def approve_retraining_decision(decision_id: str, approved_by: str):
    """Approve a pending retraining decision"""
    try:
        orchestrator = get_orchestrator()
        success = await orchestrator.approve_retraining(decision_id, approved_by)
        
        if not success:
            raise HTTPException(status_code=404, detail="Decision not found or already processed")
        
        logger.info(f"Retraining decision {decision_id} approved by {approved_by}")
        return {"message": "Decision approved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving retraining decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decisions/{decision_id}/reject")
async def reject_retraining_decision(decision_id: str, rejected_by: str):
    """Reject a pending retraining decision"""
    try:
        orchestrator = get_orchestrator()
        success = await orchestrator.reject_retraining(decision_id, rejected_by)
        
        if not success:
            raise HTTPException(status_code=404, detail="Decision not found or already processed")
        
        logger.info(f"Retraining decision {decision_id} rejected by {rejected_by}")
        return {"message": "Decision rejected successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting retraining decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[RetrainingHistoryResponse])
async def get_retraining_history(
    model_type: Optional[str] = None,
    limit: int = 50
):
    """Get retraining history"""
    try:
        if limit > 100:
            limit = 100  # Cap limit for performance
        
        model_type_enum = None
        if model_type:
            try:
                model_type_enum = ModelType(model_type)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model_type. Must be one of: {[e.value for e in ModelType]}"
                )
        
        trigger_api = get_trigger_api()
        history = await trigger_api.get_retraining_history(model_type_enum, limit)
        
        return [RetrainingHistoryResponse(**run) for run in history]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting retraining history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config", response_model=Dict[str, str])
async def update_trigger_config(request: TriggerConfigRequest):
    """Update trigger configuration"""
    try:
        trigger_api = get_trigger_api()
        
        # Convert request to dict, filtering None values
        config_updates = {k: v for k, v in request.dict().items() if v is not None}
        
        if not config_updates:
            raise HTTPException(status_code=400, detail="No configuration updates provided")
        
        success = await trigger_api.update_trigger_config(config_updates)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
        
        logger.info(f"Trigger configuration updated: {config_updates}")
        return {"message": "Configuration updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating trigger config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config", response_model=Dict[str, Any])
async def get_trigger_config():
    """Get current trigger configuration"""
    try:
        orchestrator = get_orchestrator()
        
        config = {
            "min_new_annotations": orchestrator.config.min_new_annotations,
            "min_annotation_ratio": orchestrator.config.min_annotation_ratio,
            "min_quality_drop": orchestrator.config.min_quality_drop,
            "min_iaa_drop": orchestrator.config.min_iaa_drop,
            "max_days_since_training": orchestrator.config.max_days_since_training,
            "min_days_between_training": orchestrator.config.min_days_between_training,
            "min_performance_drop": orchestrator.config.min_performance_drop,
            "novelty_threshold": orchestrator.config.novelty_threshold,
            "enable_automatic_training": orchestrator.config.enable_automatic_training,
            "require_manual_approval": orchestrator.config.require_manual_approval,
            "parallel_training_limit": orchestrator.config.parallel_training_limit,
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Error getting trigger config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-monitoring")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start the retraining orchestrator monitoring"""
    try:
        orchestrator = get_orchestrator()
        
        if orchestrator.running:
            return {"message": "Monitoring already running"}
        
        # Start monitoring in background
        background_tasks.add_task(orchestrator.start_monitoring)
        
        logger.info("Retraining orchestrator monitoring started")
        return {"message": "Monitoring started successfully"}
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-monitoring")
async def stop_monitoring():
    """Stop the retraining orchestrator monitoring"""
    try:
        orchestrator = get_orchestrator()
        orchestrator.stop_monitoring()
        
        logger.info("Retraining orchestrator monitoring stopped")
        return {"message": "Monitoring stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-types", response_model=List[str])
async def get_available_model_types():
    """Get list of available model types for retraining"""
    return [model_type.value for model_type in ModelType]

@router.get("/trigger-conditions", response_model=List[str])
async def get_trigger_conditions():
    """Get list of possible trigger conditions"""
    return [condition.value for condition in TriggerCondition]

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for retraining system"""
    try:
        orchestrator = get_orchestrator()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "orchestrator_running": orchestrator.running,
            "active_decisions_count": len(orchestrator.get_active_decisions()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
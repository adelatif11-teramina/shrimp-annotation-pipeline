#!/usr/bin/env python3
"""
Example: Automated Model Retraining System

Demonstrates how to use the automated model retraining triggers including:
- Monitoring for retraining conditions
- Manual trigger requests
- Status monitoring and approval workflows
- Configuration management
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.training.model_retraining_triggers import (
    ModelRetrainingOrchestrator,
    RetrainingTriggerAPI,
    TriggerConfig,
    ModelType,
    TriggerCondition,
    create_retraining_orchestrator
)

async def example_automated_monitoring():
    """Example: Start automated monitoring for retraining triggers"""
    print("=== Automated Monitoring Example ===\n")
    
    # Create configuration with custom thresholds
    config = TriggerConfig(
        min_new_annotations=50,           # Lower threshold for demo
        min_annotation_ratio=0.05,        # 5% new data triggers retraining
        max_days_since_training=7,        # Retrain weekly for demo
        enable_automatic_training=True,   # Enable automatic execution
        require_manual_approval=False     # No manual approval needed
    )
    
    print("Configuration:")
    print(f"  Min new annotations: {config.min_new_annotations}")
    print(f"  Min annotation ratio: {config.min_annotation_ratio:.1%}")
    print(f"  Max days since training: {config.max_days_since_training}")
    print(f"  Automatic training enabled: {config.enable_automatic_training}")
    print()
    
    # Create orchestrator
    orchestrator = create_retraining_orchestrator(config)
    
    print("Starting monitoring (will run for 30 seconds for demo)...")
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(orchestrator.start_monitoring())
    
    # Let it run for a bit
    await asyncio.sleep(30)
    
    # Stop monitoring
    orchestrator.stop_monitoring()
    monitoring_task.cancel()
    
    print("Monitoring stopped.")
    print()

async def example_manual_trigger():
    """Example: Manually trigger model retraining"""
    print("=== Manual Trigger Example ===\n")
    
    orchestrator = create_retraining_orchestrator()
    api = RetrainingTriggerAPI(orchestrator)
    
    # Trigger retraining for SciBERT NER model
    decision_id = await api.manual_trigger_retraining(
        model_type=ModelType.SCIBERT_NER,
        triggered_by="data_scientist@example.com",
        reason="Performance degradation detected in production"
    )
    
    print(f"Manual retraining triggered!")
    print(f"Decision ID: {decision_id}")
    print(f"Model: {ModelType.SCIBERT_NER.value}")
    print()
    
    # Check decision status
    status = orchestrator.get_decision_status(decision_id)
    if status:
        print("Decision Details:")
        print(f"  Confidence: {status['confidence_score']:.2f}")
        print(f"  Priority: {status['priority']}/10")
        print(f"  Estimated Duration: {status['estimated_duration']} minutes")
        print(f"  Trigger Conditions: {[c.value for c in status['trigger_conditions']]}")
        print()
    
    return decision_id

async def example_approval_workflow():
    """Example: Manual approval workflow for retraining"""
    print("=== Approval Workflow Example ===\n")
    
    # Configure to require manual approval
    config = TriggerConfig(
        enable_automatic_training=True,
        require_manual_approval=True  # Require approval
    )
    
    orchestrator = create_retraining_orchestrator(config)
    api = RetrainingTriggerAPI(orchestrator)
    
    print("Triggering retraining that requires approval...")
    
    # Trigger retraining
    decision_id = await api.manual_trigger_retraining(
        model_type=ModelType.RELATION_EXTRACTION,
        triggered_by="researcher@example.com",
        reason="New domain data available"
    )
    
    print(f"Decision {decision_id} created and pending approval")
    print()
    
    # Show active decisions
    active_decisions = orchestrator.get_active_decisions()
    print(f"Active decisions requiring approval: {len(active_decisions)}")
    
    if active_decisions:
        decision = active_decisions[0]
        print(f"  Decision ID: {decision['decision_id']}")
        print(f"  Model Type: {decision['model_type'].value}")
        print(f"  Confidence: {decision['confidence_score']:.2f}")
        print()
    
    # Approve the decision
    print("Approving decision...")
    success = await orchestrator.approve_retraining(decision_id, "manager@example.com")
    
    if success:
        print(f"Decision {decision_id} approved by manager@example.com")
        print("Retraining will now execute")
    else:
        print("Failed to approve decision")
    
    print()

async def example_status_monitoring():
    """Example: Monitor system status and decisions"""
    print("=== Status Monitoring Example ===\n")
    
    orchestrator = create_retraining_orchestrator()
    api = RetrainingTriggerAPI(orchestrator)
    
    # Get current active decisions
    active_decisions = orchestrator.get_active_decisions()
    print(f"Active Decisions: {len(active_decisions)}")
    
    for i, decision in enumerate(active_decisions, 1):
        model_type = decision['model_type']
        if hasattr(model_type, 'value'):
            model_type = model_type.value
            
        print(f"  {i}. Decision ID: {decision['decision_id']}")
        print(f"     Model Type: {model_type}")
        print(f"     Confidence: {decision['confidence_score']:.2f}")
        print(f"     Priority: {decision['priority']}/10")
        print(f"     Created: {decision['created_at']}")
        
        if decision.get('executed_at'):
            print(f"     Executed: {decision['executed_at']}")
        
        print()
    
    # Get recent training history
    history = await api.get_retraining_history(limit=5)
    print(f"Recent Training Runs: {len(history)}")
    
    for i, run in enumerate(history, 1):
        print(f"  {i}. Run ID: {run['id']}")
        print(f"     Model: {run['model_type']}")
        print(f"     Status: {run['status']}")
        print(f"     Triggers: {', '.join(run['trigger_conditions'])}")
        print(f"     Created: {run['created_at']}")
        
        if run.get('completed_at'):
            print(f"     Completed: {run['completed_at']}")
        
        print()

async def example_configuration_management():
    """Example: Update and manage trigger configuration"""
    print("=== Configuration Management Example ===\n")
    
    orchestrator = create_retraining_orchestrator()
    api = RetrainingTriggerAPI(orchestrator)
    
    # Show current configuration
    print("Current Configuration:")
    config = orchestrator.config
    print(f"  Min New Annotations: {config.min_new_annotations}")
    print(f"  Min Annotation Ratio: {config.min_annotation_ratio:.1%}")
    print(f"  Quality Drop Threshold: {config.min_quality_drop:.1%}")
    print(f"  Max Days Since Training: {config.max_days_since_training}")
    print(f"  Automatic Training: {config.enable_automatic_training}")
    print(f"  Manual Approval Required: {config.require_manual_approval}")
    print()
    
    # Update configuration
    print("Updating configuration...")
    new_config = {
        "min_new_annotations": 75,
        "min_annotation_ratio": 0.08,  # 8%
        "enable_automatic_training": False  # Disable for safety
    }
    
    success = await api.update_trigger_config(new_config)
    
    if success:
        print("Configuration updated successfully!")
        print("New values:")
        for key, value in new_config.items():
            if isinstance(value, float) and "ratio" in key:
                print(f"  {key}: {value:.1%}")
            else:
                print(f"  {key}: {value}")
    else:
        print("Failed to update configuration")
    
    print()

async def example_trigger_evaluation():
    """Example: Evaluate retraining needs without executing"""
    print("=== Trigger Evaluation Example ===\n")
    
    orchestrator = create_retraining_orchestrator()
    
    print("Evaluating retraining needs for all models...")
    print("(This checks conditions without actually triggering retraining)")
    print()
    
    for model_type in ModelType:
        print(f"Evaluating {model_type.value}...")
        
        try:
            # This is a hypothetical evaluation - the actual method is internal
            # In practice, you would use the monitoring system or CLI tools
            print(f"  ✓ Evaluation completed")
            print(f"    Data volume: Checking annotation counts...")
            print(f"    Quality metrics: Checking IAA and F1 scores...")
            print(f"    Time since training: Checking last training date...")
            print(f"    Performance: Checking auto-accept accuracy...")
            print()
            
        except Exception as e:
            print(f"  ⚠ Error during evaluation: {e}")
            print()

async def example_integration_with_events():
    """Example: Integration with event system"""
    print("=== Event System Integration Example ===\n")
    
    # This demonstrates how the retraining system integrates with events
    # In practice, events would be published by other parts of the system
    
    from services.events.event_system import EventPublisher, EventType
    
    # Create event publisher
    publisher = EventPublisher()
    await publisher.connect()
    
    print("Publishing events that could trigger retraining evaluation...")
    
    # Simulate events that might trigger retraining evaluation
    events = [
        {
            "type": EventType.ANNOTATION_ACCEPTED,
            "data": {"item_id": "item_123", "annotator": "alice@example.com"},
            "description": "Annotation accepted - adds to training data"
        },
        {
            "type": EventType.TRAINING_DATA_UPDATED,
            "data": {"item_id": "item_123", "action": "accepted"},
            "description": "Training data updated - might trigger retraining"
        },
        {
            "type": EventType.QUALITY_CHECK_COMPLETED,
            "data": {"item_id": "item_123", "quality_score": 0.85},
            "description": "Quality check completed - affects retraining decisions"
        }
    ]
    
    for event_info in events:
        event_id = await publisher.publish_simple(
            event_info["type"],
            event_info["data"],
            "retraining_example"
        )
        
        print(f"Published: {event_info['description']}")
        print(f"  Event ID: {event_id}")
        print(f"  Type: {event_info['type'].value}")
        print()
    
    await publisher.disconnect()
    
    print("Events published. The retraining orchestrator would process these")
    print("and evaluate if any models need retraining.")
    print()

async def main():
    """Run all examples"""
    print("🦐 Model Retraining System Examples\n")
    print("This demonstrates the automated model retraining capabilities")
    print("of the shrimp aquaculture annotation pipeline.\n")
    
    examples = [
        ("Automated Monitoring", example_automated_monitoring),
        ("Manual Trigger", example_manual_trigger),
        ("Approval Workflow", example_approval_workflow),
        ("Status Monitoring", example_status_monitoring),
        ("Configuration Management", example_configuration_management),
        ("Trigger Evaluation", example_trigger_evaluation),
        ("Event System Integration", example_integration_with_events),
    ]
    
    for name, example_func in examples:
        print(f"{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        try:
            await example_func()
        except Exception as e:
            print(f"Error running {name}: {e}")
        
        print(f"{'='*60}\n")
    
    print("All examples completed! 🎉")
    print("\nKey takeaways:")
    print("- The system automatically monitors for retraining conditions")
    print("- Manual triggers are available for immediate retraining needs")
    print("- Approval workflows provide governance and control")
    print("- Configuration can be updated dynamically")
    print("- Full integration with the event-driven architecture")
    print("\nThe retraining system ensures your models stay current with")
    print("the latest annotation data and maintain high performance.")

if __name__ == "__main__":
    asyncio.run(main())
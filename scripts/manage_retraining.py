#!/usr/bin/env python3
"""
CLI Script for Managing Model Retraining System

Provides command-line interface for monitoring, triggering, and configuring
the automated model retraining system.
"""

import asyncio
import argparse
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.training.model_retraining_triggers import (
    ModelRetrainingOrchestrator,
    RetrainingTriggerAPI,
    TriggerConfig,
    ModelType,
    create_retraining_orchestrator
)

def load_config(config_path: str) -> TriggerConfig:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract relevant config sections
        data_config = config_data.get('data_volume', {})
        quality_config = config_data.get('quality', {})
        schedule_config = config_data.get('schedule', {})
        performance_config = config_data.get('performance', {})
        domain_config = config_data.get('domain', {})
        system_config = config_data.get('system', {})
        
        # Create TriggerConfig instance
        config = TriggerConfig(
            min_new_annotations=data_config.get('min_new_annotations', 100),
            min_annotation_ratio=data_config.get('min_annotation_ratio', 0.1),
            min_quality_drop=quality_config.get('min_quality_drop', 0.05),
            min_iaa_drop=quality_config.get('min_iaa_drop', 0.1),
            max_days_since_training=schedule_config.get('max_days_since_training', 30),
            min_days_between_training=schedule_config.get('min_days_between_training', 7),
            min_performance_drop=performance_config.get('min_performance_drop', 0.05),
            novelty_threshold=domain_config.get('novelty_threshold', 0.3),
            enable_automatic_training=system_config.get('enable_automatic_training', True),
            require_manual_approval=system_config.get('require_manual_approval', False),
            parallel_training_limit=system_config.get('parallel_training_limit', 2)
        )
        
        return config
        
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        return TriggerConfig()
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration...")
        return TriggerConfig()

async def monitor_command(args):
    """Run the monitoring loop"""
    config = load_config(args.config)
    orchestrator = create_retraining_orchestrator(config, args.redis_url)
    
    print(f"Starting retraining orchestrator...")
    print(f"Config: automatic_training={config.enable_automatic_training}, "
          f"manual_approval={config.require_manual_approval}")
    
    try:
        await orchestrator.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down orchestrator...")
        orchestrator.stop_monitoring()
        await orchestrator.stop()

async def trigger_command(args):
    """Manually trigger retraining"""
    config = load_config(args.config)
    orchestrator = create_retraining_orchestrator(config, args.redis_url)
    api = RetrainingTriggerAPI(orchestrator)
    
    try:
        model_type = ModelType(args.model_type)
    except ValueError:
        print(f"Invalid model type: {args.model_type}")
        print(f"Available types: {[e.value for e in ModelType]}")
        return
    
    decision_id = await api.manual_trigger_retraining(
        model_type=model_type,
        triggered_by=args.user,
        reason=args.reason
    )
    
    print(f"Retraining triggered successfully!")
    print(f"Decision ID: {decision_id}")
    print(f"Model Type: {args.model_type}")
    print(f"Triggered by: {args.user}")

async def status_command(args):
    """Show system status"""
    config = load_config(args.config)
    orchestrator = create_retraining_orchestrator(config, args.redis_url)
    api = RetrainingTriggerAPI(orchestrator)
    
    print("=== Model Retraining System Status ===\n")
    
    # Show configuration
    print("Configuration:")
    print(f"  Automatic Training: {config.enable_automatic_training}")
    print(f"  Manual Approval Required: {config.require_manual_approval}")
    print(f"  Min New Annotations: {config.min_new_annotations}")
    print(f"  Min Annotation Ratio: {config.min_annotation_ratio:.2%}")
    print(f"  Max Days Since Training: {config.max_days_since_training}")
    print(f"  Parallel Training Limit: {config.parallel_training_limit}")
    print()
    
    # Show active decisions
    active_decisions = orchestrator.get_active_decisions()
    print(f"Active Decisions: {len(active_decisions)}")
    
    if active_decisions:
        for decision in active_decisions:
            model_type = decision['model_type']
            if hasattr(model_type, 'value'):
                model_type = model_type.value
            
            print(f"  Decision ID: {decision['decision_id']}")
            print(f"  Model Type: {model_type}")
            print(f"  Confidence: {decision['confidence_score']:.2f}")
            print(f"  Priority: {decision['priority']}/10")
            print(f"  Created: {decision['created_at']}")
            if decision.get('executed_at'):
                print(f"  Executed: {decision['executed_at']}")
            print()
    
    # Show recent history
    history = await api.get_retraining_history(limit=5)
    print(f"Recent Training Runs (last 5):")
    
    if history:
        for run in history:
            print(f"  Run ID: {run['id']}")
            print(f"  Model Type: {run['model_type']}")
            print(f"  Status: {run['status']}")
            print(f"  Created: {run['created_at']}")
            if run.get('completed_at'):
                print(f"  Completed: {run['completed_at']}")
            print()
    else:
        print("  No training runs found")

async def history_command(args):
    """Show training history"""
    config = load_config(args.config)
    orchestrator = create_retraining_orchestrator(config, args.redis_url)
    api = RetrainingTriggerAPI(orchestrator)
    
    model_type = None
    if args.model_type:
        try:
            model_type = ModelType(args.model_type)
        except ValueError:
            print(f"Invalid model type: {args.model_type}")
            return
    
    history = await api.get_retraining_history(model_type, args.limit)
    
    print(f"=== Training History ===")
    if args.model_type:
        print(f"Model Type: {args.model_type}")
    print(f"Showing last {len(history)} runs\n")
    
    if not history:
        print("No training runs found")
        return
    
    for i, run in enumerate(history, 1):
        print(f"{i}. Run ID: {run['id']}")
        print(f"   Model Type: {run['model_type']}")
        print(f"   Status: {run['status']}")
        print(f"   Triggers: {', '.join(run['trigger_conditions'])}")
        if run.get('confidence_score'):
            print(f"   Confidence: {run['confidence_score']:.2f}")
        print(f"   Created: {run['created_at']}")
        if run.get('completed_at'):
            print(f"   Completed: {run['completed_at']}")
        print()

async def approve_command(args):
    """Approve a pending retraining decision"""
    config = load_config(args.config)
    orchestrator = create_retraining_orchestrator(config, args.redis_url)
    
    success = await orchestrator.approve_retraining(args.decision_id, args.user)
    
    if success:
        print(f"Decision {args.decision_id} approved by {args.user}")
    else:
        print(f"Decision {args.decision_id} not found or already processed")

async def reject_command(args):
    """Reject a pending retraining decision"""
    config = load_config(args.config)
    orchestrator = create_retraining_orchestrator(config, args.redis_url)
    
    success = await orchestrator.reject_retraining(args.decision_id, args.user)
    
    if success:
        print(f"Decision {args.decision_id} rejected by {args.user}")
    else:
        print(f"Decision {args.decision_id} not found or already processed")

async def evaluate_command(args):
    """Evaluate retraining needs for all models"""
    config = load_config(args.config)
    orchestrator = create_retraining_orchestrator(config, args.redis_url)
    
    print("=== Evaluating Retraining Needs ===\n")
    
    for model_type in ModelType:
        print(f"Evaluating {model_type.value}...")
        
        try:
            decision = await orchestrator._evaluate_model_retraining(model_type)
            
            if decision:
                print(f"  ✓ Retraining recommended")
                print(f"    Confidence: {decision.confidence_score:.2f}")
                print(f"    Priority: {decision.priority}/10")
                print(f"    Triggers: {[c.value for c in decision.trigger_conditions]}")
                print(f"    Estimated Duration: {decision.estimated_duration} minutes")
            else:
                print(f"  ✗ No retraining needed")
            
        except Exception as e:
            print(f"  ⚠ Error evaluating: {e}")
        
        print()

def main():
    parser = argparse.ArgumentParser(description="Manage Model Retraining System")
    parser.add_argument('--config', '-c', 
                       default='config/retraining_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--redis-url', '-r',
                       default='redis://localhost:6379',
                       help='Redis URL for event system')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring loop')
    
    # Trigger command
    trigger_parser = subparsers.add_parser('trigger', help='Manually trigger retraining')
    trigger_parser.add_argument('model_type', choices=[e.value for e in ModelType],
                               help='Model type to retrain')
    trigger_parser.add_argument('--user', '-u', required=True,
                               help='User triggering the retraining')
    trigger_parser.add_argument('--reason', '-m', required=True,
                               help='Reason for manual trigger')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show training history')
    history_parser.add_argument('--model-type', choices=[e.value for e in ModelType],
                               help='Filter by model type')
    history_parser.add_argument('--limit', type=int, default=20,
                               help='Maximum number of runs to show')
    
    # Approve command
    approve_parser = subparsers.add_parser('approve', help='Approve pending decision')
    approve_parser.add_argument('decision_id', help='Decision ID to approve')
    approve_parser.add_argument('--user', '-u', required=True,
                               help='User approving the decision')
    
    # Reject command
    reject_parser = subparsers.add_parser('reject', help='Reject pending decision')
    reject_parser.add_argument('decision_id', help='Decision ID to reject')
    reject_parser.add_argument('--user', '-u', required=True,
                              help='User rejecting the decision')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate retraining needs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Command dispatch
    commands = {
        'monitor': monitor_command,
        'trigger': trigger_command,
        'status': status_command,
        'history': history_command,
        'approve': approve_command,
        'reject': reject_command,
        'evaluate': evaluate_command,
    }
    
    if args.command in commands:
        asyncio.run(commands[args.command](args))
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
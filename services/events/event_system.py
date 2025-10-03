"""
Event Streaming System

Implements event-driven architecture for the annotation pipeline using Redis Streams.
Handles events for document ingestion, candidate generation, annotation decisions, and training triggers.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import redis
import redis.asyncio as aioredis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for the annotation pipeline"""
    # Document events
    DOCUMENT_INGESTED = "document.ingested"
    DOCUMENT_PROCESSED = "document.processed"
    
    # Candidate events
    CANDIDATES_GENERATED = "candidates.generated"
    CANDIDATES_STAGED = "candidates.staged"
    
    # Triage events
    ITEM_TRIAGED = "triage.item_added"
    ITEM_ASSIGNED = "triage.item_assigned"
    ITEM_COMPLETED = "triage.item_completed"
    
    # Annotation events
    ANNOTATION_STARTED = "annotation.started"
    ANNOTATION_ACCEPTED = "annotation.accepted"
    ANNOTATION_REJECTED = "annotation.rejected"
    ANNOTATION_MODIFIED = "annotation.modified"
    
    # Auto-accept events
    AUTO_ACCEPT_TRIGGERED = "auto_accept.triggered"
    AUTO_ACCEPT_APPLIED = "auto_accept.applied"
    
    # Quality events
    QUALITY_CHECK_COMPLETED = "quality.check_completed"
    IAA_CALCULATED = "quality.iaa_calculated"
    
    # Training events
    TRAINING_DATA_UPDATED = "training.data_updated"
    MODEL_TRAINING_TRIGGERED = "training.model_triggered"
    MODEL_TRAINING_COMPLETED = "training.model_completed"
    
    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    ERROR_OCCURRED = "system.error"

@dataclass
class Event:
    """Base event structure"""
    event_id: str
    event_type: EventType
    timestamp: str
    source: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EventPublisher:
    """Event publisher using Redis Streams"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize event publisher.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client = None
        self.stream_prefix = "annotation_pipeline"
        
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        logger.info("Connected to Redis for event publishing")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def publish(self, event: Event) -> str:
        """
        Publish event to Redis Stream.
        
        Args:
            event: Event to publish
            
        Returns:
            Event ID from Redis
        """
        if not self.redis_client:
            await self.connect()
        
        stream_name = f"{self.stream_prefix}:{event.event_type.value}"
        
        # Convert event to Redis fields
        fields = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "source": event.source,
            "data": json.dumps(event.data),
            "correlation_id": event.correlation_id or "",
            "metadata": json.dumps(event.metadata or {})
        }
        
        try:
            redis_id = await self.redis_client.xadd(stream_name, fields)
            logger.debug(f"Published event {event.event_id} to stream {stream_name}")
            return redis_id
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            raise
    
    async def publish_simple(self, 
                           event_type: EventType,
                           data: Dict[str, Any],
                           source: str,
                           correlation_id: Optional[str] = None) -> str:
        """
        Publish a simple event.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            correlation_id: Optional correlation ID
            
        Returns:
            Event ID
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            source=source,
            data=data,
            correlation_id=correlation_id
        )
        
        return await self.publish(event)

class EventConsumer:
    """Event consumer using Redis Streams"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 consumer_group: str = "annotation_workers",
                 consumer_name: Optional[str] = None):
        """
        Initialize event consumer.
        
        Args:
            redis_url: Redis connection URL
            consumer_group: Consumer group name
            consumer_name: Consumer name (defaults to UUID)
        """
        self.redis_url = redis_url
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or str(uuid.uuid4())
        self.redis_client = None
        self.stream_prefix = "annotation_pipeline"
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.running = False
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        logger.info(f"Connected to Redis for event consumption as {self.consumer_name}")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """
        Register event handler.
        
        Args:
            event_type: Event type to handle
            handler: Async function to handle the event
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")
    
    async def create_consumer_groups(self):
        """Create consumer groups for all event streams"""
        if not self.redis_client:
            await self.connect()
        
        for event_type in EventType:
            stream_name = f"{self.stream_prefix}:{event_type.value}"
            try:
                await self.redis_client.xgroup_create(
                    stream_name, 
                    self.consumer_group, 
                    id="0", 
                    mkstream=True
                )
                logger.debug(f"Created consumer group for {stream_name}")
            except ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Consumer group already exists
                    pass
                else:
                    logger.error(f"Failed to create consumer group for {stream_name}: {e}")
    
    async def consume_events(self, batch_size: int = 10, block_time: int = 1000):
        """
        Start consuming events from all registered streams.
        
        Args:
            batch_size: Number of events to read in each batch
            block_time: Block time in milliseconds
        """
        if not self.redis_client:
            await self.connect()
        
        await self.create_consumer_groups()
        
        # Build streams list for registered handlers
        streams = {}
        for event_type in self.handlers.keys():
            stream_name = f"{self.stream_prefix}:{event_type.value}"
            streams[stream_name] = ">"
        
        if not streams:
            logger.warning("No event handlers registered")
            return
        
        self.running = True
        logger.info(f"Starting event consumption for {len(streams)} streams")
        
        while self.running:
            try:
                # Read from multiple streams
                result = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    streams,
                    count=batch_size,
                    block=block_time
                )
                
                # Process events
                for stream_name, events in result:
                    for event_id, fields in events:
                        await self._process_event(stream_name, event_id, fields)
                        
            except Exception as e:
                logger.error(f"Error consuming events: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, stream_name: str, event_id: str, fields: Dict[str, str]):
        """Process a single event"""
        try:
            # Parse event
            event_type_str = fields.get("event_type")
            event_type = EventType(event_type_str)
            
            event = Event(
                event_id=fields.get("event_id"),
                event_type=event_type,
                timestamp=fields.get("timestamp"),
                source=fields.get("source"),
                data=json.loads(fields.get("data", "{}")),
                correlation_id=fields.get("correlation_id") or None,
                metadata=json.loads(fields.get("metadata", "{}"))
            )
            
            # Execute handlers
            if event_type in self.handlers:
                for handler in self.handlers[event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Handler error for {event.event_id}: {e}")
            
            # Acknowledge event
            await self.redis_client.xack(stream_name, self.consumer_group, event_id)
            logger.debug(f"Processed event {event.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to process event {event_id}: {e}")
    
    def stop(self):
        """Stop consuming events"""
        self.running = False

class EventOrchestrator:
    """Orchestrates the complete event-driven workflow"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize event orchestrator.
        
        Args:
            redis_url: Redis connection URL
        """
        self.publisher = EventPublisher(redis_url)
        self.consumer = EventConsumer(redis_url, "orchestrator", "main_orchestrator")
        self.workflow_handlers = {}
        self._setup_workflow_handlers()
    
    def _setup_workflow_handlers(self):
        """Setup workflow event handlers"""
        # Document workflow
        self.consumer.register_handler(EventType.DOCUMENT_INGESTED, self._handle_document_ingested)
        
        # Candidate workflow
        self.consumer.register_handler(EventType.CANDIDATES_GENERATED, self._handle_candidates_generated)
        
        # Annotation workflow
        self.consumer.register_handler(EventType.ANNOTATION_ACCEPTED, self._handle_annotation_accepted)
        self.consumer.register_handler(EventType.ANNOTATION_MODIFIED, self._handle_annotation_modified)
        
        # Training workflow
        self.consumer.register_handler(EventType.TRAINING_DATA_UPDATED, self._handle_training_data_updated)
    
    async def start(self):
        """Start the orchestrator"""
        await self.publisher.connect()
        await self.consumer.connect()
        
        # Start consuming events
        await self.consumer.consume_events()
    
    async def stop(self):
        """Stop the orchestrator"""
        self.consumer.stop()
        await self.publisher.disconnect()
        await self.consumer.disconnect()
    
    # Workflow handlers
    async def _handle_document_ingested(self, event: Event):
        """Handle document ingestion completion"""
        doc_id = event.data.get("doc_id")
        logger.info(f"Document {doc_id} ingested, triggering candidate generation")
        
        # Trigger candidate generation
        await self.publisher.publish_simple(
            EventType.CANDIDATES_GENERATED,
            {"doc_id": doc_id, "trigger": "document_ingested"},
            "orchestrator",
            event.correlation_id
        )
    
    async def _handle_candidates_generated(self, event: Event):
        """Handle candidate generation completion"""
        doc_id = event.data.get("doc_id")
        candidate_count = event.data.get("candidate_count", 0)
        
        logger.info(f"Generated {candidate_count} candidates for {doc_id}, adding to triage")
        
        # Add to triage queue
        await self.publisher.publish_simple(
            EventType.ITEM_TRIAGED,
            {"doc_id": doc_id, "candidate_count": candidate_count},
            "orchestrator",
            event.correlation_id
        )
    
    async def _handle_annotation_accepted(self, event: Event):
        """Handle annotation acceptance"""
        item_id = event.data.get("item_id")
        annotator = event.data.get("annotator")
        
        logger.info(f"Annotation {item_id} accepted by {annotator}")
        
        # Update training data
        await self.publisher.publish_simple(
            EventType.TRAINING_DATA_UPDATED,
            {"item_id": item_id, "action": "accepted"},
            "orchestrator",
            event.correlation_id
        )
        
        # Trigger quality check
        await self.publisher.publish_simple(
            EventType.QUALITY_CHECK_COMPLETED,
            {"item_id": item_id, "status": "accepted"},
            "orchestrator",
            event.correlation_id
        )
    
    async def _handle_annotation_modified(self, event: Event):
        """Handle annotation modification"""
        item_id = event.data.get("item_id")
        
        # Update training data with modified annotation
        await self.publisher.publish_simple(
            EventType.TRAINING_DATA_UPDATED,
            {"item_id": item_id, "action": "modified"},
            "orchestrator",
            event.correlation_id
        )
    
    async def _handle_training_data_updated(self, event: Event):
        """Handle training data update"""
        action = event.data.get("action")
        
        # Check if we should trigger model retraining
        # This would typically check thresholds, schedules, etc.
        if self._should_trigger_training():
            await self.publisher.publish_simple(
                EventType.MODEL_TRAINING_TRIGGERED,
                {"trigger_reason": f"training_data_{action}"},
                "orchestrator",
                event.correlation_id
            )
    
    def _should_trigger_training(self) -> bool:
        """Determine if model training should be triggered"""
        # Implement training trigger logic here
        # For example: check if enough new annotations accumulated
        return False  # Placeholder

class EventMetrics:
    """Event system metrics and monitoring"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.stream_prefix = "annotation_pipeline"
    
    def get_stream_info(self, event_type: EventType) -> Dict[str, Any]:
        """Get information about a specific stream"""
        stream_name = f"{self.stream_prefix}:{event_type.value}"
        try:
            info = self.redis_client.xinfo_stream(stream_name)
            return {
                "stream_name": stream_name,
                "length": info.get("length", 0),
                "last_generated_id": info.get("last-generated-id"),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
            }
        except ResponseError:
            return {"stream_name": stream_name, "length": 0, "exists": False}
    
    def get_consumer_group_info(self, event_type: EventType, group_name: str) -> Dict[str, Any]:
        """Get consumer group information"""
        stream_name = f"{self.stream_prefix}:{event_type.value}"
        try:
            groups = self.redis_client.xinfo_groups(stream_name)
            for group in groups:
                if group.get("name") == group_name:
                    return group
            return {"exists": False}
        except ResponseError:
            return {"exists": False}
    
    def get_all_streams_summary(self) -> Dict[str, Any]:
        """Get summary of all event streams"""
        summary = {}
        for event_type in EventType:
            summary[event_type.value] = self.get_stream_info(event_type)
        return summary

# Factory functions for easy setup
def create_publisher(redis_url: str = "redis://localhost:6379") -> EventPublisher:
    """Create an event publisher"""
    return EventPublisher(redis_url)

def create_consumer(consumer_group: str, 
                   consumer_name: Optional[str] = None,
                   redis_url: str = "redis://localhost:6379") -> EventConsumer:
    """Create an event consumer"""
    return EventConsumer(redis_url, consumer_group, consumer_name)

def create_orchestrator(redis_url: str = "redis://localhost:6379") -> EventOrchestrator:
    """Create an event orchestrator"""
    return EventOrchestrator(redis_url)

# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Event system utilities")
    parser.add_argument("--action", choices=["publish", "consume", "orchestrate", "metrics"], required=True)
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--event-type", choices=[e.value for e in EventType])
    parser.add_argument("--data", help="Event data as JSON string")
    parser.add_argument("--consumer-group", default="test_group")
    
    args = parser.parse_args()
    
    async def main():
        if args.action == "publish":
            if not args.event_type or not args.data:
                print("--event-type and --data required for publish")
                return
            
            publisher = create_publisher(args.redis_url)
            await publisher.connect()
            
            event_type = EventType(args.event_type)
            data = json.loads(args.data)
            
            event_id = await publisher.publish_simple(
                event_type, data, "cli_test"
            )
            print(f"Published event: {event_id}")
            
            await publisher.disconnect()
        
        elif args.action == "consume":
            consumer = create_consumer(args.consumer_group, "cli_consumer", args.redis_url)
            
            # Register simple handler
            async def simple_handler(event: Event):
                print(f"Received event: {event.event_type.value} - {event.data}")
            
            for event_type in EventType:
                consumer.register_handler(event_type, simple_handler)
            
            await consumer.consume_events()
        
        elif args.action == "orchestrate":
            orchestrator = create_orchestrator(args.redis_url)
            await orchestrator.start()
        
        elif args.action == "metrics":
            metrics = EventMetrics(args.redis_url)
            summary = metrics.get_all_streams_summary()
            print(json.dumps(summary, indent=2))
    
    asyncio.run(main())
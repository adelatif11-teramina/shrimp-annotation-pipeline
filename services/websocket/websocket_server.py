"""
WebSocket Server for Real-time Collaborative Annotation
Handles real-time updates, user presence, and collaboration features
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# WebSocket connection manager
@dataclass
class UserConnection:
    websocket: WebSocket
    user_id: str
    username: str
    role: str
    connected_at: datetime
    current_item_id: Optional[str] = None

class WebSocketManager:
    def __init__(self):
        # Active connections by user_id
        self.active_connections: Dict[str, UserConnection] = {}
        # Users working on specific items
        self.item_assignments: Dict[str, Set[str]] = {}
        # Broadcast channels for different events
        self.channels: Dict[str, Set[str]] = {
            "annotations": set(),
            "triage_updates": set(),
            "user_presence": set(),
            "system_alerts": set()
        }
    
    async def connect(self, websocket: WebSocket, user_id: str, username: str, role: str):
        """Connect a new user"""
        await websocket.accept()
        
        connection = UserConnection(
            websocket=websocket,
            user_id=user_id,
            username=username,
            role=role,
            connected_at=datetime.utcnow()
        )
        
        self.active_connections[user_id] = connection
        
        # Subscribe to all channels by default
        for channel in self.channels:
            self.channels[channel].add(user_id)
        
        # Notify others of new user
        await self.broadcast_user_presence()
        
        # Send current state to new user
        await self.send_current_state(user_id)
    
    def disconnect(self, user_id: str):
        """Disconnect a user"""
        if user_id in self.active_connections:
            connection = self.active_connections[user_id]
            
            # Remove from item assignments
            if connection.current_item_id:
                self.remove_user_from_item(user_id, connection.current_item_id)
            
            # Remove from all channels
            for channel in self.channels:
                self.channels[channel].discard(user_id)
            
            # Remove connection
            del self.active_connections[user_id]
    
    async def send_personal_message(self, user_id: str, message: dict):
        """Send message to specific user"""
        if user_id in self.active_connections:
            connection = self.active_connections[user_id]
            try:
                await connection.websocket.send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"Failed to send message to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast_to_channel(self, channel: str, message: dict, exclude_user: str = None):
        """Broadcast message to all users in a channel"""
        if channel not in self.channels:
            return
        
        message["channel"] = channel
        message["timestamp"] = datetime.utcnow().isoformat()
        
        disconnected_users = []
        
        for user_id in self.channels[channel]:
            if exclude_user and user_id == exclude_user:
                continue
                
            if user_id in self.active_connections:
                try:
                    await self.active_connections[user_id].websocket.send_text(json.dumps(message))
                except Exception as e:
                    logging.error(f"Failed to broadcast to {user_id}: {e}")
                    disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            self.disconnect(user_id)
    
    async def broadcast_user_presence(self):
        """Broadcast current user presence to all connected users"""
        users = []
        for user_id, connection in self.active_connections.items():
            users.append({
                "user_id": user_id,
                "username": connection.username,
                "role": connection.role,
                "current_item": connection.current_item_id,
                "connected_at": connection.connected_at.isoformat()
            })
        
        message = {
            "type": "user_presence_update",
            "users": users,
            "total_users": len(users)
        }
        
        await self.broadcast_to_channel("user_presence", message)
    
    async def send_current_state(self, user_id: str):
        """Send current system state to newly connected user"""
        # Get active assignments
        active_assignments = {}
        for item_id, users in self.item_assignments.items():
            if users:
                active_assignments[item_id] = list(users)
        
        message = {
            "type": "initial_state",
            "active_assignments": active_assignments,
            "connected_users": len(self.active_connections)
        }
        
        await self.send_personal_message(user_id, message)
    
    def assign_user_to_item(self, user_id: str, item_id: str):
        """Assign user to work on specific item"""
        if user_id in self.active_connections:
            # Remove from previous item
            connection = self.active_connections[user_id]
            if connection.current_item_id:
                self.remove_user_from_item(user_id, connection.current_item_id)
            
            # Assign to new item
            connection.current_item_id = item_id
            if item_id not in self.item_assignments:
                self.item_assignments[item_id] = set()
            self.item_assignments[item_id].add(user_id)
    
    def remove_user_from_item(self, user_id: str, item_id: str):
        """Remove user from item assignment"""
        if item_id in self.item_assignments:
            self.item_assignments[item_id].discard(user_id)
            if not self.item_assignments[item_id]:
                del self.item_assignments[item_id]
        
        if user_id in self.active_connections:
            self.active_connections[user_id].current_item_id = None
    
    def get_users_on_item(self, item_id: str) -> List[str]:
        """Get list of users working on specific item"""
        return list(self.item_assignments.get(item_id, set()))

# Global WebSocket manager
manager = WebSocketManager()

# FastAPI app for WebSocket server
app = FastAPI(title="Annotation WebSocket Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def websocket_info():
    """WebSocket server info"""
    return {
        "service": "annotation_websocket",
        "status": "running",
        "connected_users": len(manager.active_connections),
        "active_assignments": len(manager.item_assignments),
        "channels": list(manager.channels.keys())
    }

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, username: str = "Anonymous", role: str = "annotator"):
    """Main WebSocket endpoint for real-time communication"""
    
    await manager.connect(websocket, user_id, username, role)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(user_id, message)
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
        await manager.broadcast_user_presence()
    except Exception as e:
        logging.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)

async def handle_websocket_message(user_id: str, message: dict):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "annotation_start":
        # User started working on an item
        item_id = message.get("item_id")
        if item_id:
            manager.assign_user_to_item(user_id, item_id)
            
            # Notify others
            await manager.broadcast_to_channel("annotations", {
                "type": "annotation_start",
                "user_id": user_id,
                "username": manager.active_connections[user_id].username,
                "item_id": item_id
            }, exclude_user=user_id)
            
            await manager.broadcast_user_presence()
    
    elif message_type == "annotation_complete":
        # User completed annotation
        item_id = message.get("item_id")
        decision = message.get("decision")
        
        if item_id:
            manager.remove_user_from_item(user_id, item_id)
            
            # Notify others
            await manager.broadcast_to_channel("annotations", {
                "type": "annotation_complete",
                "user_id": user_id,
                "username": manager.active_connections[user_id].username,
                "item_id": item_id,
                "decision": decision
            }, exclude_user=user_id)
            
            await manager.broadcast_user_presence()
    
    elif message_type == "annotation_progress":
        # User is making progress on annotation
        item_id = message.get("item_id")
        progress = message.get("progress", {})
        
        await manager.broadcast_to_channel("annotations", {
            "type": "annotation_progress",
            "user_id": user_id,
            "username": manager.active_connections[user_id].username,
            "item_id": item_id,
            "progress": progress
        }, exclude_user=user_id)
    
    elif message_type == "triage_update":
        # Triage queue was updated
        update_type = message.get("update_type")
        affected_items = message.get("items", [])
        
        await manager.broadcast_to_channel("triage_updates", {
            "type": "triage_update",
            "update_type": update_type,
            "items": affected_items,
            "updated_by": user_id
        }, exclude_user=user_id)
    
    elif message_type == "system_alert":
        # System-wide alert (admin only)
        if manager.active_connections[user_id].role == "admin":
            alert_message = message.get("message")
            alert_level = message.get("level", "info")
            
            await manager.broadcast_to_channel("system_alerts", {
                "type": "system_alert",
                "message": alert_message,
                "level": alert_level,
                "from_user": manager.active_connections[user_id].username
            })
    
    elif message_type == "heartbeat":
        # Keep connection alive
        await manager.send_personal_message(user_id, {
            "type": "heartbeat_response",
            "server_time": datetime.utcnow().isoformat()
        })

@app.get("/status")
async def get_status():
    """Get current WebSocket server status"""
    return {
        "connected_users": len(manager.active_connections),
        "active_assignments": {
            item_id: [manager.active_connections[uid].username for uid in users if uid in manager.active_connections]
            for item_id, users in manager.item_assignments.items()
        },
        "users": [
            {
                "user_id": uid,
                "username": conn.username,
                "role": conn.role,
                "current_item": conn.current_item_id,
                "connected_duration": (datetime.utcnow() - conn.connected_at).total_seconds()
            }
            for uid, conn in manager.active_connections.items()
        ]
    }

# HTTP endpoints for WebSocket integration
@app.post("/broadcast/annotation")
async def broadcast_annotation_update(update: dict):
    """Broadcast annotation update to all connected users"""
    await manager.broadcast_to_channel("annotations", {
        "type": "external_annotation_update",
        **update
    })
    return {"status": "broadcasted"}

@app.post("/broadcast/triage")
async def broadcast_triage_update(update: dict):
    """Broadcast triage queue update to all connected users"""
    await manager.broadcast_to_channel("triage_updates", {
        "type": "external_triage_update",
        **update
    })
    return {"status": "broadcasted"}

@app.post("/alert")
async def send_system_alert(alert: dict):
    """Send system-wide alert"""
    await manager.broadcast_to_channel("system_alerts", {
        "type": "system_alert",
        "message": alert.get("message"),
        "level": alert.get("level", "info"),
        "from_system": True
    })
    return {"status": "alert_sent"}

if __name__ == "__main__":
    print("🔗 Starting WebSocket Server for Real-time Collaboration")
    print("📡 WebSocket endpoint: ws://localhost:8001/ws/{user_id}")
    print("📊 Status endpoint: http://localhost:8001/status")
    
    uvicorn.run(
        "websocket_server:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )
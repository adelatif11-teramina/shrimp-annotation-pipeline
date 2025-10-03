import { useState, useEffect, useRef, useCallback } from 'react';

const useWebSocket = (userId, username = 'Anonymous', role = 'annotator') => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [connectedUsers, setConnectedUsers] = useState([]);
  const [activeAssignments, setActiveAssignments] = useState({});
  const [lastMessage, setLastMessage] = useState(null);
  const [systemAlerts, setSystemAlerts] = useState([]);
  
  const ws = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectTimeout = useRef(null);
  
  const connect = useCallback(() => {
    try {
      const wsUrl = `ws://localhost:8002/ws/${userId}?username=${encodeURIComponent(username)}&role=${role}`;
      console.log('🔗 Connecting to WebSocket:', wsUrl);
      
      ws.current = new WebSocket(wsUrl);
      setConnectionStatus('connecting');
      
      ws.current.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        
        // Send heartbeat every 30 seconds
        const heartbeatInterval = setInterval(() => {
          if (ws.current?.readyState === WebSocket.OPEN) {
            sendMessage({
              type: 'heartbeat',
              timestamp: new Date().toISOString()
            });
          } else {
            clearInterval(heartbeatInterval);
          }
        }, 30000);
      };
      
      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLastMessage(message);
          handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      ws.current.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        
        // Attempt to reconnect
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
          console.log(`🔄 Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`);
          
          reconnectTimeout.current = setTimeout(() => {
            reconnectAttempts.current++;
            setConnectionStatus('reconnecting');
            connect();
          }, delay);
        } else {
          console.error('❌ Max reconnection attempts reached');
          setConnectionStatus('failed');
        }
      };
      
      ws.current.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setConnectionStatus('error');
      };
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('error');
    }
  }, [userId, username, role]);
  
  const handleMessage = (message) => {
    console.log('📨 WebSocket message:', message);
    
    switch (message.type) {
      case 'user_presence_update':
        setConnectedUsers(message.users || []);
        break;
        
      case 'initial_state':
        setActiveAssignments(message.active_assignments || {});
        break;
        
      case 'annotation_start':
        setActiveAssignments(prev => ({
          ...prev,
          [message.item_id]: [...(prev[message.item_id] || []), message.user_id]
        }));
        break;
        
      case 'annotation_complete':
        setActiveAssignments(prev => {
          const updated = { ...prev };
          if (updated[message.item_id]) {
            updated[message.item_id] = updated[message.item_id].filter(id => id !== message.user_id);
            if (updated[message.item_id].length === 0) {
              delete updated[message.item_id];
            }
          }
          return updated;
        });
        break;
        
      case 'system_alert':
        const alert = {
          id: Date.now(),
          message: message.message,
          level: message.level || 'info',
          timestamp: message.timestamp,
          from_user: message.from_user,
          from_system: message.from_system
        };
        
        setSystemAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
        
        // Auto-remove non-critical alerts after 5 seconds
        if (alert.level !== 'error' && alert.level !== 'critical') {
          setTimeout(() => {
            setSystemAlerts(prev => prev.filter(a => a.id !== alert.id));
          }, 5000);
        }
        break;
        
      case 'heartbeat_response':
        // Server is alive, no action needed
        break;
        
      default:
        console.log('Unhandled message type:', message.type);
    }
  };
  
  const sendMessage = useCallback((message) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
      return false;
    }
  }, []);
  
  const startAnnotation = useCallback((itemId) => {
    return sendMessage({
      type: 'annotation_start',
      item_id: itemId,
      timestamp: new Date().toISOString()
    });
  }, [sendMessage]);
  
  const completeAnnotation = useCallback((itemId, decision, details = {}) => {
    return sendMessage({
      type: 'annotation_complete',
      item_id: itemId,
      decision: decision,
      details: details,
      timestamp: new Date().toISOString()
    });
  }, [sendMessage]);
  
  const updateProgress = useCallback((itemId, progress) => {
    return sendMessage({
      type: 'annotation_progress',
      item_id: itemId,
      progress: progress,
      timestamp: new Date().toISOString()
    });
  }, [sendMessage]);
  
  const notifyTriageUpdate = useCallback((updateType, items) => {
    return sendMessage({
      type: 'triage_update',
      update_type: updateType,
      items: items,
      timestamp: new Date().toISOString()
    });
  }, [sendMessage]);
  
  const sendSystemAlert = useCallback((message, level = 'info') => {
    return sendMessage({
      type: 'system_alert',
      message: message,
      level: level,
      timestamp: new Date().toISOString()
    });
  }, [sendMessage]);
  
  const dismissAlert = useCallback((alertId) => {
    setSystemAlerts(prev => prev.filter(alert => alert.id !== alertId));
  }, []);
  
  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
    }
    
    if (ws.current) {
      ws.current.close(1000, 'User disconnected');
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
    reconnectAttempts.current = 0;
  }, []);
  
  // Connect on mount - temporarily disabled
  useEffect(() => {
    // Temporarily disable WebSocket for demo
    console.log('WebSocket disabled for demo mode');
    // if (userId) {
    //   connect();
    // }
    
    return () => {
      // disconnect();
    };
  }, [userId, connect, disconnect]);
  
  // Check if user is working on specific item
  const isUserOnItem = useCallback((itemId, checkUserId = null) => {
    const targetUserId = checkUserId || userId;
    return activeAssignments[itemId]?.includes(targetUserId) || false;
  }, [activeAssignments, userId]);
  
  // Get users working on specific item
  const getUsersOnItem = useCallback((itemId) => {
    const userIds = activeAssignments[itemId] || [];
    return userIds.map(id => 
      connectedUsers.find(user => user.user_id === id)
    ).filter(Boolean);
  }, [activeAssignments, connectedUsers]);
  
  // Get collaboration conflicts (multiple users on same item)
  const getCollaborationConflicts = useCallback(() => {
    return Object.entries(activeAssignments)
      .filter(([itemId, users]) => users.length > 1)
      .map(([itemId, userIds]) => ({
        itemId,
        users: userIds.map(id => 
          connectedUsers.find(user => user.user_id === id)
        ).filter(Boolean)
      }));
  }, [activeAssignments, connectedUsers]);
  
  return {
    // Connection state
    isConnected,
    connectionStatus,
    
    // User presence
    connectedUsers,
    activeAssignments,
    
    // Messages and alerts
    lastMessage,
    systemAlerts,
    
    // Actions
    startAnnotation,
    completeAnnotation,
    updateProgress,
    notifyTriageUpdate,
    sendSystemAlert,
    dismissAlert,
    
    // Utilities
    isUserOnItem,
    getUsersOnItem,
    getCollaborationConflicts,
    
    // Connection control
    connect,
    disconnect,
    sendMessage
  };
};

export default useWebSocket;
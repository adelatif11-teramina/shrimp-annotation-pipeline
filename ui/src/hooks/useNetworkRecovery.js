import { useState, useEffect, useRef, useCallback } from 'react';
import { useAnnotationAPI } from './useAnnotationAPI';

const RETRY_DELAYS = [1000, 2000, 5000, 10000, 30000]; // Progressive backoff
const MAX_RETRIES = 5;
const NETWORK_CHECK_INTERVAL = 10000; // 10 seconds

/**
 * Network recovery hook for handling API failures gracefully
 * Provides retry logic, offline detection, and queued operations
 */
export function useNetworkRecovery() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [connectionStatus, setConnectionStatus] = useState('online'); // online, offline, reconnecting
  const [failedOperations, setFailedOperations] = useState([]);
  const [retryCount, setRetryCount] = useState(0);
  
  const { apiCall } = useAnnotationAPI();
  const networkCheckIntervalRef = useRef(null);
  const retryTimeoutRef = useRef(null);

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setConnectionStatus('reconnecting');
      console.debug('🌐 Network connection restored');
      
      // Retry failed operations after a brief delay
      setTimeout(() => {
        retryFailedOperations();
      }, 1000);
    };

    const handleOffline = () => {
      setIsOnline(false);
      setConnectionStatus('offline');
      console.debug('📡 Network connection lost');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Periodic connectivity check
  useEffect(() => {
    const checkConnectivity = async () => {
      try {
        // Simple ping to check if API is reachable
        await apiCall('/api/health', { timeout: 5000 });
        
        if (connectionStatus === 'offline' || connectionStatus === 'reconnecting') {
          setConnectionStatus('online');
          setRetryCount(0);
          console.debug('✅ API connectivity confirmed');
        }
      } catch (error) {
        if (connectionStatus === 'online') {
          setConnectionStatus('offline');
          console.warn('❌ API connectivity lost');
        }
      }
    };

    networkCheckIntervalRef.current = setInterval(checkConnectivity, NETWORK_CHECK_INTERVAL);

    return () => {
      if (networkCheckIntervalRef.current) {
        clearInterval(networkCheckIntervalRef.current);
      }
    };
  }, [connectionStatus, apiCall]);

  // Add failed operation to queue
  const queueFailedOperation = useCallback((operation) => {
    const queuedOp = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      retryCount: 0,
      ...operation
    };

    setFailedOperations(prev => {
      // Avoid duplicates for the same operation
      const exists = prev.some(op => 
        op.type === operation.type && 
        op.itemId === operation.itemId
      );
      
      if (exists) {
        return prev.map(op => 
          op.type === operation.type && op.itemId === operation.itemId
            ? { ...op, data: operation.data, timestamp: queuedOp.timestamp }
            : op
        );
      }
      
      return [...prev, queuedOp];
    });

    console.debug('📥 Queued failed operation:', operation.type, operation.itemId);
  }, []);

  // Execute a single operation with retry logic
  const executeWithRetry = useCallback(async (operation) => {
    const { type, endpoint, method, data, itemId } = operation;
    
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const result = await apiCall(endpoint, {
          method: method || 'POST',
          data,
          timeout: 10000
        });

        console.debug(`✅ Successfully executed ${type} for item ${itemId} (attempt ${attempt + 1})`);
        return result;
        
      } catch (error) {
        console.debug(`❌ Attempt ${attempt + 1} failed for ${type}:`, error.message);
        
        if (attempt < MAX_RETRIES - 1) {
          const delay = RETRY_DELAYS[attempt] || RETRY_DELAYS[RETRY_DELAYS.length - 1];
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          throw error;
        }
      }
    }
  }, [apiCall]);

  // Retry all failed operations
  const retryFailedOperations = useCallback(async () => {
    if (failedOperations.length === 0) {
      setConnectionStatus('online');
      return;
    }

    setConnectionStatus('reconnecting');
    const succeededOperations = [];
    const stillFailedOperations = [];

    for (const operation of failedOperations) {
      try {
        await executeWithRetry(operation);
        succeededOperations.push(operation.id);
      } catch (error) {
        console.warn(`Failed to retry operation ${operation.type}:`, error);
        stillFailedOperations.push({
          ...operation,
          retryCount: operation.retryCount + 1
        });
      }
    }

    // Update failed operations list
    setFailedOperations(stillFailedOperations);

    if (stillFailedOperations.length === 0) {
      setConnectionStatus('online');
      console.info('🎉 All failed operations successfully retried');
    } else {
      console.warn(`⚠️ ${stillFailedOperations.length} operations still failed after retry`);
    }
  }, [failedOperations, executeWithRetry]);

  // Wrapper for API calls with automatic retry and queuing
  const callWithRecovery = useCallback(async (endpoint, options = {}, operationInfo = {}) => {
    const { type = 'api_call', itemId = 'unknown' } = operationInfo;

    try {
      return await apiCall(endpoint, options);
    } catch (error) {
      console.debug('🔄 API call failed, handling recovery:', error.message);

      // Check if it's a network error that should be retried
      const isNetworkError = 
        error.code === 'NETWORK_ERROR' ||
        error.code === 'ECONNREFUSED' ||
        error.message.includes('Network Error') ||
        error.message.includes('fetch') ||
        !navigator.onLine;

      const isServerError = error.response?.status >= 500;

      if (isNetworkError || isServerError) {
        // Queue the operation for retry
        queueFailedOperation({
          type,
          endpoint,
          method: options.method || 'GET',
          data: options.data,
          itemId,
          originalError: error.message
        });

        setConnectionStatus('offline');

        // For critical operations, try immediate retry once
        if (operationInfo.critical) {
          try {
            await new Promise(resolve => setTimeout(resolve, 2000));
            return await apiCall(endpoint, options);
          } catch (retryError) {
            console.warn('Immediate retry also failed');
          }
        }

        // Return a special error to indicate the operation was queued
        const queuedError = new Error('Operation queued for retry due to network issue');
        queuedError.queued = true;
        queuedError.originalError = error;
        throw queuedError;
      }

      // For non-network errors, throw immediately
      throw error;
    }
  }, [apiCall, queueFailedOperation]);

  // Clear specific failed operation
  const clearFailedOperation = useCallback((operationId) => {
    setFailedOperations(prev => prev.filter(op => op.id !== operationId));
  }, []);

  // Clear all failed operations
  const clearAllFailedOperations = useCallback(() => {
    setFailedOperations([]);
  }, []);

  // Manual retry trigger
  const retryNow = useCallback(() => {
    if (failedOperations.length > 0) {
      retryFailedOperations();
    }
  }, [failedOperations, retryFailedOperations]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (networkCheckIntervalRef.current) {
        clearInterval(networkCheckIntervalRef.current);
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, []);

  return {
    // State
    isOnline,
    connectionStatus,
    failedOperations,
    hasFailedOperations: failedOperations.length > 0,
    
    // Actions
    callWithRecovery,
    retryFailedOperations,
    retryNow,
    clearFailedOperation,
    clearAllFailedOperations,
    
    // Helpers
    isConnected: connectionStatus === 'online',
    isReconnecting: connectionStatus === 'reconnecting'
  };
}

export default useNetworkRecovery;

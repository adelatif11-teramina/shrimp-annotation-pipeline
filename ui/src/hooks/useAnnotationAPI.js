import { useState } from 'react';
import axios from 'axios';

// Use relative URL to leverage the proxy configuration in package.json
// This avoids CORS issues by making requests same-origin
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

// Create axios instance with proper defaults
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Get token from localStorage or use default for local development
const getAuthToken = () => {
  const token = localStorage.getItem('auth_token') || 'local-admin-2024';
  console.log('🔑 Auth token:', token);
  return token;
};

export function useAnnotationAPI() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiCall = async (url, options = {}) => {
    setIsLoading(true);
    setError(null);
    
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getAuthToken()}`,
      ...options.headers,
    };
    
    console.log('🚀 API Call:', url, { headers, ...options });
    console.log('📋 Headers object:', JSON.stringify(headers, null, 2));
    
    try {
      const response = await axiosInstance({
        url,
        headers,
        ...options,
      });
      return response.data;
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentItem = async (itemId) => {
    return apiCall(`/api/triage/queue?limit=50`).then(response => {
      const items = response.items || response;
      // Convert itemId to number for comparison since API returns numeric IDs
      const numericId = parseInt(itemId, 10);
      return items.find(item => 
        item.item_id === itemId || 
        item.item_id === numericId || 
        item.id === itemId || 
        item.id === numericId
      ) || null;
    });
  };

  const getNextItem = async () => {
    return apiCall('/api/triage/next').then(response => response.item || null);
  };

  const submitAnnotation = async (annotation) => {
    return apiCall('/api/annotations/decide', {
      method: 'POST',
      data: annotation,
    });
  };

  const skipItem = async (itemId) => {
    // Mark item as skipped - would need API endpoint
    return apiCall(`/api/triage/items/${itemId}/skip`, {
      method: 'POST',
    });
  };

  const generateCandidates = async (sentenceData) => {
    return apiCall('/api/candidates/generate', {
      method: 'POST',
      data: sentenceData,
    });
  };

  const getTriageQueue = async (filters = {}) => {
    const params = new URLSearchParams(filters);
    return apiCall(`/api/triage/queue?${params}`);
  };

  const getTriageStatistics = async () => {
    return apiCall('/api/triage/statistics');
  };

  const getSystemStatistics = async () => {
    return apiCall('/api/statistics/overview');
  };

  const exportGoldData = async (exportRequest) => {
    return apiCall('/api/export/gold', {
      method: 'POST',
      data: exportRequest,
    });
  };

  const getDocuments = async (filters = {}) => {
    const params = new URLSearchParams(filters);
    return apiCall(`/api/documents?${params}`);
  };

  const ingestDocument = async (documentData) => {
    return apiCall('/api/documents/ingest', {
      method: 'POST',
      data: documentData,
    });
  };

  // New annotation results functions
  const getAnnotations = async (filters = {}) => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value && value !== 'all') {
        params.append(key, value);
      }
    });
    return apiCall(`/api/annotations?${params}`);
  };

  const getAnnotationDetail = async (annotationId) => {
    return apiCall(`/api/annotations/${annotationId}`);
  };

  const getAnnotationStatistics = async () => {
    return apiCall('/api/annotations/statistics');
  };

  const exportAnnotations = async (filters = {}, format = 'json') => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value && value !== 'all') {
        params.append(key, value);
      }
    });
    params.append('format', format);
    
    // For exports, we need to handle file downloads differently
    const url = `/api/annotations/export?${params}`;
    window.open(url, '_blank');
    return { success: true };
  };

  return {
    getCurrentItem,
    getNextItem,
    submitAnnotation,
    skipItem,
    generateCandidates,
    getTriageQueue,
    getTriageStatistics,
    getSystemStatistics,
    exportGoldData,
    getDocuments,
    ingestDocument,
    getAnnotations,
    getAnnotationDetail,
    getAnnotationStatistics,
    exportAnnotations,
    apiCall, // Export apiCall for direct use
    isLoading,
    error,
  };
}

export default useAnnotationAPI;
import { useCallback, useMemo, useRef, useState } from 'react';
import axios from 'axios';

// Resolve the base URL once so we can respect runtime overrides
const resolveBaseUrl = () => {
  try {
    const storedSettings = localStorage.getItem('annotation_settings');
    if (storedSettings) {
      const parsed = JSON.parse(storedSettings);
      if (parsed?.api_url) {
        return parsed.api_url;
      }
    }
  } catch (error) {
    console.warn('Failed to read annotation settings:', error);
  }

  return process.env.REACT_APP_API_URL || '';
};

// Lazily create the axios instance so that the latest base URL is applied
const createAxiosInstance = () =>
  axios.create({
    baseURL: resolveBaseUrl(),
    headers: {
      'Content-Type': 'application/json',
    },
  });

// Get token from localStorage or use default for local development
const getAuthToken = () => localStorage.getItem('auth_token') || 'local-admin-2024';

export function useAnnotationAPI() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const axiosRef = useRef(null);
  const inFlightRequests = useRef(0);

  const axiosInstance = useMemo(() => {
    if (!axiosRef.current) {
      axiosRef.current = createAxiosInstance();
    }
    return axiosRef.current;
  }, []);

  const updateLoadingState = useCallback((delta) => {
    inFlightRequests.current += delta;
    if (inFlightRequests.current < 0) {
      inFlightRequests.current = 0;
    }
    setIsLoading(inFlightRequests.current > 0);
  }, []);

  const apiCall = useCallback(
    async (url, options = {}) => {
      updateLoadingState(1);
      setError(null);

      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${getAuthToken()}`,
        ...options.headers,
      };

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
        updateLoadingState(-1);
      }
    },
    [axiosInstance, updateLoadingState],
  );

  const getCurrentItem = useCallback(
    async (itemId) => {
      return apiCall(`/api/triage/queue?limit=1000&sort_by=priority`).then((response) => {
        const items = response.items || response;
        // Convert itemId to number for comparison since API returns numeric IDs
        const numericId = parseInt(itemId, 10);
        const item = items.find(
          (item) =>
            item.item_id === itemId ||
            item.item_id === numericId ||
            item.id === itemId ||
            item.id === numericId,
        );
        
        if (!item) {
          console.log(`🔍 Item ${itemId} not found. Available items:`, 
            items.slice(0, 5).map(i => ({ id: i.item_id || i.id, text: i.text?.substring(0, 50) })));
          console.log(`🔍 [DEBUG] Total items received: ${items.length}`);
          console.log(`🔍 [DEBUG] Full response:`, response);
        }
        
        return item || null;
      });
    },
    [apiCall],
  );

  const getNextItem = useCallback(async () => {
    return apiCall('/api/triage/next').then((response) => response.item || null);
  }, [apiCall]);

  const submitAnnotation = useCallback(
    async (annotation) => {
      return apiCall('/api/annotations/decide', {
        method: 'POST',
        data: annotation,
      });
    },
    [apiCall],
  );

  const skipItem = useCallback(
    async (itemId) => {
      // Mark item as skipped - would need API endpoint
      return apiCall(`/api/triage/items/${itemId}/skip`, {
        method: 'POST',
      });
    },
    [apiCall],
  );

  const generateCandidates = useCallback(
    async (sentenceData) => {
      return apiCall('/api/candidates/generate', {
        method: 'POST',
        data: sentenceData,
      });
    },
    [apiCall],
  );

  const getTriageQueue = useCallback(
    async (filters = {}) => {
      const params = new URLSearchParams(filters);
      return apiCall(`/api/triage/queue?${params}`);
    },
    [apiCall],
  );

  const getTriageStatistics = useCallback(async () => {
    return apiCall('/api/triage/statistics');
  }, [apiCall]);

  const getSystemStatistics = useCallback(async () => {
    return apiCall('/api/statistics/overview');
  }, [apiCall]);

  const exportGoldData = useCallback(
    async (exportRequest) => {
      return apiCall('/api/export/gold', {
        method: 'POST',
        data: exportRequest,
      });
    },
    [apiCall],
  );

  const getDocuments = useCallback(
    async (filters = {}) => {
      const params = new URLSearchParams(filters);
      return apiCall(`/api/documents?${params}`);
    },
    [apiCall],
  );

  const ingestDocument = useCallback(
    async (documentData) => {
      return apiCall('/api/documents/ingest', {
        method: 'POST',
        data: documentData,
      });
    },
    [apiCall],
  );

  // New annotation results functions
  const getAnnotations = useCallback(
    async (filters = {}) => {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value && value !== 'all') {
          params.append(key, value);
        }
      });
      return apiCall(`/api/annotations?${params}`);
    },
    [apiCall],
  );

  const getAnnotationDetail = useCallback(
    async (annotationId) => {
      return apiCall(`/api/annotations/${annotationId}`);
    },
    [apiCall],
  );

  const getAnnotationStatistics = useCallback(async () => {
    return apiCall('/api/annotations/statistics');
  }, [apiCall]);

  const exportAnnotations = useCallback(
    async (filters = {}, format = 'json') => {
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
    },
    [apiCall],
  );

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

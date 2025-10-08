import { useState, useEffect, useRef, useCallback } from 'react';
import { useAnnotationAPI } from './useAnnotationAPI';

const AUTO_SAVE_INTERVAL = 30000; // 30 seconds
const DRAFT_KEY_PREFIX = 'annotation_draft_';

/**
 * Auto-save hook for annotation work in progress
 * Prevents data loss from network issues, browser crashes, etc.
 */
export function useAutoSave(itemId, initialData = null, userSettings = {}) {
  const [isDrafted, setIsDrafted] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);
  const [saveStatus, setSaveStatus] = useState('idle'); // idle, saving, saved, error
  const { apiCall } = useAnnotationAPI();
  
  const autoSaveIntervalRef = useRef(null);
  const currentDataRef = useRef(null);
  const lastSavedDataRef = useRef(null);

  const getDraftKey = (id) => `${DRAFT_KEY_PREFIX}${id}`;

  // Load existing draft on mount
  useEffect(() => {
    if (itemId) {
      const draftKey = getDraftKey(itemId);
      const savedDraft = localStorage.getItem(draftKey);
      
      if (savedDraft) {
        try {
          const draftData = JSON.parse(savedDraft);
          
          // Check if draft is expired based on retention settings
          const draftAge = Date.now() - new Date(draftData.timestamp).getTime();
          const retentionMs = (userSettings.draft_retention_days || 7) * 24 * 60 * 60 * 1000;
          
          if (draftAge > retentionMs) {
            console.debug('🗑️ Draft expired, removing:', itemId);
            localStorage.removeItem(draftKey);
            return;
          }
          
          setIsDrafted(true);
          setLastSaved(new Date(draftData.timestamp));
          console.debug('📄 Loaded existing draft for item:', itemId);
        } catch (error) {
          console.warn('Failed to parse draft data:', error);
          localStorage.removeItem(draftKey);
        }
      }
    }
  }, [itemId, userSettings.draft_retention_days]);

  // Check if data has changed since last save
  const hasChanges = useCallback(() => {
    if (!currentDataRef.current || !lastSavedDataRef.current) {
      return !!currentDataRef.current;
    }
    
    return JSON.stringify(currentDataRef.current) !== JSON.stringify(lastSavedDataRef.current);
  }, []);

  // Save draft to localStorage
  const saveDraftLocally = useCallback((data) => {
    if (!itemId || !data) return;

    const draftKey = getDraftKey(itemId);
    const draftData = {
      itemId,
      timestamp: new Date().toISOString(),
      data
    };

    try {
      localStorage.setItem(draftKey, JSON.stringify(draftData));
      setIsDrafted(true);
      setLastSaved(new Date());
      console.debug('💾 Saved draft locally for item:', itemId);
    } catch (error) {
      console.error('Failed to save draft locally:', error);
    }
  }, [itemId]);

  // Save draft to server (optional backup)
  const saveDraftToServer = useCallback(async (data) => {
    if (!itemId || !data) return;

    try {
      setSaveStatus('saving');
      
      await apiCall('/api/annotations/draft', {
        method: 'POST',
        data: {
          item_id: itemId,
          draft_data: data,
          timestamp: new Date().toISOString()
        }
      });

      setSaveStatus('saved');
      lastSavedDataRef.current = { ...data };
      console.debug('☁️ Saved draft to server for item:', itemId);
      
    } catch (error) {
      console.warn('Failed to save draft to server:', error);
      setSaveStatus('error');
      // Don't throw - local save is the critical part
    }
  }, [itemId, apiCall]);

  // Auto-save function
  const autoSave = useCallback(async (data, forceLocal = false) => {
    if (!data || !hasChanges()) return;

    currentDataRef.current = data;

    // Always save locally for immediate protection
    saveDraftLocally(data);

    // Also try to save to server unless explicitly local-only
    if (!forceLocal) {
      await saveDraftToServer(data);
    }
  }, [hasChanges, saveDraftLocally, saveDraftToServer]);

  // Start auto-save interval
  const startAutoSave = useCallback((data) => {
    if (autoSaveIntervalRef.current) {
      clearInterval(autoSaveIntervalRef.current);
    }

    currentDataRef.current = data;

    autoSaveIntervalRef.current = setInterval(() => {
      if (currentDataRef.current && hasChanges()) {
        autoSave(currentDataRef.current);
      }
    }, AUTO_SAVE_INTERVAL);

    console.debug('🔄 Started auto-save for item:', itemId);
  }, [itemId, hasChanges, autoSave]);

  // Stop auto-save interval
  const stopAutoSave = useCallback(() => {
    if (autoSaveIntervalRef.current) {
      clearInterval(autoSaveIntervalRef.current);
      autoSaveIntervalRef.current = null;
      console.debug('⏹️ Stopped auto-save for item:', itemId);
    }
  }, [itemId]);

  // Load draft data
  const loadDraft = useCallback(() => {
    if (!itemId) return null;

    const draftKey = getDraftKey(itemId);
    const savedDraft = localStorage.getItem(draftKey);

    if (savedDraft) {
      try {
        const draftData = JSON.parse(savedDraft);
        return draftData.data;
      } catch (error) {
        console.warn('Failed to load draft:', error);
        return null;
      }
    }

    return null;
  }, [itemId]);

  // Clear draft (after successful submission)
  const clearDraft = useCallback(() => {
    if (!itemId) return;

    const draftKey = getDraftKey(itemId);
    localStorage.removeItem(draftKey);
    setIsDrafted(false);
    setLastSaved(null);
    setSaveStatus('idle');
    
    // Also clear from server
    apiCall('/api/annotations/draft', {
      method: 'DELETE',
      data: { item_id: itemId }
    }).catch(error => {
      console.warn('Failed to clear server draft:', error);
    });

    console.debug('🗑️ Cleared draft for item:', itemId);
  }, [itemId, apiCall]);

  // Update current data reference
  const updateData = useCallback((data) => {
    currentDataRef.current = data;
  }, []);

  // Manual save (for immediate saves on important changes)
  const saveNow = useCallback(async (data) => {
    if (data) {
      currentDataRef.current = data;
    }
    
    if (currentDataRef.current) {
      await autoSave(currentDataRef.current);
    }
  }, [autoSave]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAutoSave();
    };
  }, [stopAutoSave]);

  // Handle page unload (save before leaving)
  useEffect(() => {
    const handleBeforeUnload = (event) => {
      if (currentDataRef.current && hasChanges()) {
        // Force local save before page unload
        saveDraftLocally(currentDataRef.current);
        
        // Show warning if there are unsaved changes
        event.preventDefault();
        event.returnValue = 'You have unsaved annotation work. Are you sure you want to leave?';
        return event.returnValue;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [hasChanges, saveDraftLocally]);

  // Auto-handle draft based on user settings
  const shouldShowDraftDialog = () => {
    return isDrafted && 
           userSettings.show_draft_dialog !== false && 
           userSettings.draft_behavior === 'ask';
  };

  const shouldAutoRestore = () => {
    return isDrafted && userSettings.draft_behavior === 'auto_restore';
  };

  const shouldAutoDiscard = () => {
    return isDrafted && userSettings.draft_behavior === 'auto_discard';
  };

  return {
    // State
    isDrafted,
    lastSaved,
    saveStatus,
    
    // Actions
    startAutoSave,
    stopAutoSave,
    loadDraft,
    clearDraft,
    updateData,
    saveNow,
    
    // Status helpers
    hasUnsavedChanges: hasChanges,
    isAutoSaving: saveStatus === 'saving',
    
    // Draft behavior helpers
    shouldShowDraftDialog,
    shouldAutoRestore,
    shouldAutoDiscard
  };
}

export default useAutoSave;

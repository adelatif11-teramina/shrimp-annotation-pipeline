import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  LinearProgress,
  Snackbar,
  Alert,
} from '@mui/material';
import { useAnnotationAPI } from '../hooks/useAnnotationAPI';
import useWebSocket from '../hooks/useWebSocket';
import useAutoSave from '../hooks/useAutoSave';
import useNetworkRecovery from '../hooks/useNetworkRecovery';
import { SystemAlerts, CollaborationConflicts } from '../components/RealTimeIndicators';
import { ENTITY_TYPES, RELATION_TYPES } from '../constants/ontology';
import AnnotationHeader from '../components/annotation/AnnotationHeader';
import AnnotationMainPanel from '../components/annotation/AnnotationMainPanel';
import AnnotationSidebar from '../components/annotation/AnnotationSidebar';
import AnnotationGuidelinesDialog from '../components/annotation/AnnotationGuidelinesDialog';
import AnnotationShortcutsDialog from '../components/annotation/AnnotationShortcutsDialog';
import useAnnotationShortcuts from '../hooks/useAnnotationShortcuts';

function AnnotationWorkspace() {
  const { itemId } = useParams();
  const navigate = useNavigate();
  
  // State management
  const [currentItem, setCurrentItem] = useState(null);
  const [entities, setEntities] = useState([]);
  const [relations, setRelations] = useState([]);
  const [topics, setTopics] = useState([]);
  const [annotationMode, setAnnotationMode] = useState('entity'); // entity, relation, topic
  const [selectedEntityType, setSelectedEntityType] = useState('SPECIES');
  const [notes, setNotes] = useState('');
  const [confidence, setConfidence] = useState('high');
  const [showGuidelines, setShowGuidelines] = useState(false);
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const currentUserId = useMemo(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('user_id') || 'anonymous';
    }
    return 'anonymous';
  }, []);

  const showFeedback = useCallback((message, severity = 'info') => {
    setFeedback({ message, severity });
  }, []);

  const handleFeedbackClose = useCallback(() => {
    setFeedback(null);
  }, []);
  
  
  // Session analytics
  const [sessionStats, setSessionStats] = useState({
    itemsCompleted: 0,
    itemsSkipped: 0,
    startTime: Date.now(),
    totalTimeSpent: 0,
    averageTimePerItem: 0
  });

  // Error recovery and auto-save hooks
  const {
    isDrafted,
    startAutoSave,
    stopAutoSave,
    loadDraft,
    clearDraft,
    updateData,
    saveNow,
    saveStatus
  } = useAutoSave(currentItem?.id || currentItem?.item_id);

  const {
    connectionStatus,
    callWithRecovery,
    hasFailedOperations,
    retryNow
  } = useNetworkRecovery();
  const [currentItemStartTime, setCurrentItemStartTime] = useState(Date.now());
  
  // WebSocket connection for real-time collaboration
  const {
    isConnected,
    connectionStatus: wsConnectionStatus,
    connectedUsers,
    activeAssignments,
    systemAlerts,
    startAnnotation,
    completeAnnotation,
    updateProgress,
    dismissAlert,
    getUsersOnItem,
    getCollaborationConflicts
  } = useWebSocket(
    localStorage.getItem('user_id') || 'anonymous',
    localStorage.getItem('username') || 'Anonymous',
    localStorage.getItem('user_role') || 'annotator'
  );

  // API hooks
  const { 
    getCurrentItem, 
    submitAnnotation, 
    getNextItem,
    skipItem,
    isLoading 
  } = useAnnotationAPI();

  // Load current item
  useEffect(() => {
    if (itemId) {
      loadItem(itemId);
    } else {
      loadNextItem();
    }
  }, [itemId]);

  // Auto-save current annotation data
  useEffect(() => {
    if (currentItem) {
      const currentData = {
        entities,
        relations,
        topics,
        notes,
        confidence
      };
      updateData(currentData);
    }
  }, [entities, relations, topics, notes, confidence, currentItem, updateData]);

  // Handle draft restoration - automatically restore without dialog
  useEffect(() => {
    if (currentItem && isDrafted) {
      handleRestoreDraft();
    }
  }, [currentItem, isDrafted]);

  // Start auto-save when annotation begins
  useEffect(() => {
    if (currentItem) {
      const initialData = {
        entities,
        relations,
        topics,
        notes,
        confidence
      };
      startAutoSave(initialData);
      
      return () => {
        stopAutoSave();
      };
    }
  }, [currentItem]);

  // Handle draft restoration
  const handleRestoreDraft = () => {
    const draft = loadDraft();
    if (draft) {
      setEntities(draft.entities || []);
      setRelations(draft.relations || []);
      setTopics(draft.topics || []);
      setNotes(draft.notes || '');
      setConfidence(draft.confidence || 'high');
    }
  };

  const loadItem = async (id) => {
    setLoading(true);
    try {
      const item = await getCurrentItem(id);
      
      if (!item) {
        console.warn(`Item with ID ${id} not found, loading next available item`);
        const nextItem = await getNextItem();
        setCurrentItem(nextItem);
        
        // Load existing entities from the queue item format
        if (nextItem && nextItem.entities) {
          // Map entities to the expected format
          const mappedEntities = nextItem.entities.map(entity => ({
            id: `entity_${nextItem.id}_${entity.id}`,
            text: entity.text,
            label: entity.label,
            type: entity.label, // Alias for compatibility
            start: entity.start,
            end: entity.end,
            confidence: entity.confidence || 0.9
          }));
          setEntities(mappedEntities);
          setRelations(nextItem.relations || []);
          setTopics(nextItem.topics || []);
        }
        // Navigate to the correct URL if we loaded a different item
        if (nextItem && nextItem.id) {
          navigate(`/annotate/${nextItem.id}`, { replace: true });
        }
        return;
      }
      
      setCurrentItem(item);
      
      // Load existing entities from the queue item format
      if (item && item.entities) {
        // Map entities to the expected format
        const mappedEntities = item.entities.map(entity => ({
          id: `entity_${item.id}_${entity.id}`,
          text: entity.text,
          label: entity.label,
          type: entity.label, // Alias for compatibility
          start: entity.start,
          end: entity.end,
          confidence: entity.confidence || 0.9
        }));
        setEntities(mappedEntities);
        setRelations(item.relations || []);
        setTopics(item.topics || []);
      }
    } catch (error) {
      console.error('Failed to load item:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateSessionStats = (action) => {
    const timeSpent = Date.now() - currentItemStartTime;
    setSessionStats(prev => {
      const newStats = { ...prev };
      newStats.totalTimeSpent += timeSpent;
      
      if (action === 'completed') {
        newStats.itemsCompleted += 1;
      } else if (action === 'skipped') {
        newStats.itemsSkipped += 1;
      }
      
      const totalItems = newStats.itemsCompleted + newStats.itemsSkipped;
      newStats.averageTimePerItem = totalItems > 0 ? newStats.totalTimeSpent / totalItems : 0;
      
      return newStats;
    });
    setCurrentItemStartTime(Date.now());
  };

  const loadNextItem = useCallback(async () => {
    setLoading(true);
    try {
      const item = await getNextItem();
      if (item) {
        navigate(`/annotate/${item.id}`);
      } else {
        navigate('/triage');
      }
    } catch (error) {
      console.error('Failed to load next item:', error);
    } finally {
      setLoading(false);
    }
  }, [getNextItem, navigate]);

  // Text selection handler
  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    if (selection.rangeCount > 0 && annotationMode === 'entity') {
      const range = selection.getRangeAt(0);
      const text = selection.toString().trim();
      
      if (text && currentItem) {
        const container = document.getElementById('sentence-text');
        if (container && container.contains(range.commonAncestorContainer)) {
          // Calculate offsets relative to sentence
          const start = range.startOffset;
          const end = range.endOffset;
          
          // Create new entity
          const newEntity = {
            id: `e${entities.length}`,
            text: text,
            label: selectedEntityType,
            start: start,
            end: end,
            confidence: 0.9,
            source: 'manual'
          };
          
          setEntities(prev => [...prev, newEntity]);
          selection.removeAllRanges();
        }
      }
    }
  }, [annotationMode, selectedEntityType, entities, currentItem]);

  // Annotation handlers
  const handleAccept = useCallback(async () => {
    if (!currentItem) {
      return;
    }

    const annotation = {
      item_id: currentItem.item_id || currentItem.id,
      candidate_id: currentItem.candidate_id || currentItem.item_id || currentItem.id,
      decision: 'accept',
      entities,
      relations,
      topics,
      confidence: confidence === 'high' ? 0.9 : confidence === 'medium' ? 0.7 : 0.5,
      notes,
      user_id: 1,
    };

    try {
      const response = await callWithRecovery(
        '/api/annotations/decide',
        {
          method: 'POST',
          data: annotation,
        },
        {
          type: 'submit_annotation',
          itemId: currentItem.id,
          critical: true,
        },
      );

      updateSessionStats('completed');
      clearDraft();
      stopAutoSave();

      if (response.next_item) {
        navigate(`/annotate/${response.next_item.item_id || response.next_item.id}`);
      } else {
        navigate('/triage');
      }
      showFeedback('Annotation accepted', 'success');
    } catch (error) {
      console.error('Failed to submit annotation:', error);
      if (error.queued) {
        showFeedback('Network issue: annotation queued for retry', 'warning');
      } else {
        showFeedback('Failed to submit annotation. Please try again.', 'error');
      }
    }
  }, [
    callWithRecovery,
    clearDraft,
    confidence,
    currentItem,
    entities,
    navigate,
    notes,
    relations,
    showFeedback,
    stopAutoSave,
    topics,
    updateSessionStats,
  ]);

  const handleReject = useCallback(async () => {
    if (!currentItem) {
      return;
    }

    const annotation = {
      item_id: currentItem.item_id || currentItem.id,
      candidate_id: currentItem.candidate_id || currentItem.item_id || currentItem.id,
      decision: 'reject',
      notes,
      user_id: 1,
    };

    try {
      const response = await submitAnnotation(annotation);
      updateSessionStats('completed');

      if (response.next_item) {
        navigate(`/annotate/${response.next_item.item_id || response.next_item.id}`);
      } else {
        navigate('/triage');
      }
      showFeedback('Annotation rejected', 'info');
    } catch (error) {
      console.error('Failed to reject annotation:', error);
      showFeedback('Failed to reject annotation. Please try again.', 'error');
    }
  }, [currentItem, navigate, notes, showFeedback, submitAnnotation, updateSessionStats]);

  const handleModify = useCallback(async () => {
    if (!currentItem) {
      return;
    }

    const annotation = {
      candidate_id: currentItem.candidate_id,
      decision: 'modified',
      final_annotation: {
        entities,
        relations,
        topics,
        confidence,
        notes,
      },
      annotator: 'current-user@example.com',
    };

    try {
      await submitAnnotation(annotation);
      updateSessionStats('completed');
      showFeedback('Annotation modified', 'success');
      await loadNextItem();
    } catch (error) {
      console.error('Failed to submit modification:', error);
      showFeedback('Failed to modify annotation. Please try again.', 'error');
    }
  }, [
    confidence,
    currentItem,
    entities,
    loadNextItem,
    notes,
    relations,
    showFeedback,
    submitAnnotation,
    topics,
    updateSessionStats,
  ]);

  const handleSave = useCallback(async () => {
    const data = {
      entities,
      relations,
      topics,
      notes,
      confidence,
    };

    try {
      await saveNow(data);
      showFeedback('Draft saved', 'success');
    } catch (error) {
      console.error('Failed to save draft:', error);
      showFeedback('Failed to save draft', 'error');
    }
  }, [confidence, entities, notes, relations, saveNow, showFeedback, topics]);

  const handleSkip = useCallback(async () => {
    if (!currentItem) {
      return;
    }

    const annotation = {
      item_id: currentItem.item_id || currentItem.id,
      candidate_id: currentItem.candidate_id || currentItem.item_id || currentItem.id,
      decision: 'skip',
      notes: 'Skipped by user',
      user_id: 1,
    };

    try {
      const response = await submitAnnotation(annotation);
      updateSessionStats('skipped');

      if (response.next_item) {
        navigate(`/annotate/${response.next_item.item_id || response.next_item.id}`);
      } else {
        navigate('/triage');
      }
      showFeedback('Annotation skipped', 'info');
    } catch (error) {
      console.error('Failed to skip item:', error);
      showFeedback('Failed to skip item. Please try again.', 'error');
    }
  }, [currentItem, navigate, showFeedback, submitAnnotation, updateSessionStats]);

  useAnnotationShortcuts({
    onAccept: handleAccept,
    onReject: handleReject,
    onModify: handleModify,
    onSkip: handleSkip,
    onSave: handleSave,
    onNext: loadNextItem,
    onSetMode: setAnnotationMode,
    onSetEntityType: setSelectedEntityType,
    openGuidelines: () => setShowGuidelines(true),
    openKeyboardHelp: () => setShowKeyboardHelp(true),
    closeKeyboardHelp: () => setShowKeyboardHelp(false),
  });

  const usersOnItem = useMemo(
    () => (currentItem ? getUsersOnItem(currentItem.id) : []),
    [currentItem, getUsersOnItem],
  );

  if (loading || !currentItem) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading annotation item...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <SystemAlerts alerts={systemAlerts} onDismiss={dismissAlert} />
      <CollaborationConflicts
        conflicts={getCollaborationConflicts()}
        onResolve={(itemId) => console.debug('Resolving conflict for item:', itemId)}
      />

      <AnnotationHeader
        currentItem={currentItem}
        sessionStats={sessionStats}
        isConnected={isConnected}
        connectionStatus={connectionStatus}
        usersOnItem={usersOnItem}
        currentUserId={currentUserId}
        onShowGuidelines={() => setShowGuidelines(true)}
        onShowKeyboardHelp={() => setShowKeyboardHelp(true)}
      />

      <Grid container spacing={3} sx={{ flexGrow: 1 }}>
        <Grid item xs={12} md={8}>
          <AnnotationMainPanel
            currentItem={currentItem}
            annotationMode={annotationMode}
            onChangeMode={setAnnotationMode}
            onTextSelection={handleTextSelection}
            entities={entities}
            setEntities={setEntities}
            selectedEntityType={selectedEntityType}
            setSelectedEntityType={setSelectedEntityType}
            relations={relations}
            setRelations={setRelations}
            topics={topics}
            setTopics={setTopics}
            entityTypes={ENTITY_TYPES}
            relationTypes={RELATION_TYPES}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <AnnotationSidebar
            currentItem={currentItem}
            confidence={confidence}
            onConfidenceChange={setConfidence}
            notes={notes}
            onNotesChange={setNotes}
            isSubmitting={isLoading}
            onAccept={handleAccept}
            onModify={handleModify}
            onReject={handleReject}
            onSkip={handleSkip}
          />
        </Grid>
      </Grid>

      <AnnotationGuidelinesDialog
        open={showGuidelines}
        onClose={() => setShowGuidelines(false)}
        entityTypes={ENTITY_TYPES}
      />

      <AnnotationShortcutsDialog
        open={showKeyboardHelp}
        onClose={() => setShowKeyboardHelp(false)}
      />

      <Snackbar
        open={Boolean(feedback)}
        autoHideDuration={4000}
        onClose={handleFeedbackClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {feedback && (
          <Alert
            severity={feedback.severity}
            onClose={handleFeedbackClose}
            sx={{ width: '100%' }}
          >
            {feedback.message}
          </Alert>
        )}
      </Snackbar>
    </Box>
  );
}

export default AnnotationWorkspace;

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  Box,
  Paper,
  Typography,
  Button,
  ButtonGroup,
  Chip,
  Grid,
  Card,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Check as AcceptIcon,
  Close as RejectIcon,
  Edit as EditIcon,
  Undo as UndoIcon,
  Redo as RedoIcon,
  Save as SaveIcon,
  SkipNext as SkipIcon,
  Help as HelpIcon,
  Keyboard as KeyboardIcon,
} from '@mui/icons-material';

import EntityAnnotator from '../components/EntityAnnotator';
import RelationAnnotator from '../components/RelationAnnotator';
import TopicAnnotator from '../components/TopicAnnotator';
import AnnotationHistory from '../components/AnnotationHistory';
import { useAnnotationAPI } from '../hooks/useAnnotationAPI';
import useWebSocket from '../hooks/useWebSocket';
import useAutoSave from '../hooks/useAutoSave';
import useNetworkRecovery from '../hooks/useNetworkRecovery';
import {
  ConnectionStatus,
  CollaborationIndicator,
  SystemAlerts,
  CollaborationConflicts
} from '../components/RealTimeIndicators';

// Entity type configurations - v2.0 Ontology (23 types)
const ENTITY_TYPES = {
  // Core biological entities
  SPECIES: { color: '#FF6B6B', description: 'Organism names (shrimp species, bacteria)' },
  PATHOGEN: { color: '#FF8E53', description: 'Disease agents (Vibrio, viruses, parasites)' },
  DISEASE: { color: '#FFD93D', description: 'Clinical conditions/syndromes (AHPND, WSD, TPD)' },
  CLINICAL_SYMPTOM: { color: '#FF4444', description: 'Observed abnormalities (white feces, lethargy)' },
  PHENOTYPIC_TRAIT: { color: '#E91E63', description: 'Measurable performance (survival rate, growth rate)' },
  GENE: { color: '#9B59B6', description: 'Genetic markers (PvIGF, hemocyanin, TLR)' },
  TREATMENT: { color: '#6BCF7F', description: 'Medications, probiotics, interventions' },
  LIFE_STAGE: { color: '#00BCD4', description: 'Development stages (PL15, juvenile, broodstock)' },
  
  // Reified entities
  MEASUREMENT: { color: '#795548', description: 'Reified measurements (28°C, 10 mg/kg, 85%)' },
  SAMPLE: { color: '#8BC34A', description: 'Physical samples with IDs (S-001, water sample W123)' },
  TEST_TYPE: { color: '#03A9F4', description: 'Diagnostic methods (PCR, qPCR, histopathology)' },
  TEST_RESULT: { color: '#009688', description: 'Test outcomes (WSSV positive, Ct=22)' },
  
  // Operational entities
  MANAGEMENT_PRACTICE: { color: '#4D96FF', description: 'Farming practices (biosecurity, water exchange)' },
  ENVIRONMENTAL_PARAM: { color: '#607D8B', description: 'Environmental factors (temperature, salinity, DO)' },
  LOCATION: { color: '#9E9E9E', description: 'Geographic/facility locations (pond, hatchery, Thailand)' },
  EVENT: { color: '#FF9800', description: 'Timestamped occurrences (mortality event, outbreak)' },
  TISSUE: { color: '#E1BEE7', description: 'Anatomical parts (hepatopancreas, gill, hemolymph)' },
  
  // Supply chain entities
  PRODUCT: { color: '#FFC107', description: 'Commercial products (PL batch, frozen shrimp)' },
  SUPPLY_ENTITY: { color: '#CDDC39', description: 'Supply chain participants (hatchery, feed supplier)' },
  PERSON: { color: '#F48FB1', description: 'Individuals (farmer, technician, veterinarian)' },
  ORGANIZATION: { color: '#CE93D8', description: 'Companies/institutions (CP Foods, research institute)' },
  
  // Procedural entities
  PROTOCOL: { color: '#90CAF9', description: 'SOPs (biosecurity protocol, treatment regimen)' },
  CERTIFICATION: { color: '#A5D6A7', description: 'Quality certs (BAP, ASC, organic)' },
};

// Relation types - v2.0 Ontology (22 types)
const RELATION_TYPES = [
  // Co-reference types (for duplicate entities)
  'same_as', 'refers_to', 'abbreviation_of', 'synonym_of', 'alias_of',
  
  // Core biological relations (v2.0)
  'infected_by',           // SPECIES → PATHOGEN
  'causes',                // [PATHOGEN|ENVIRONMENTAL_PARAM] → [DISEASE|CLINICAL_SYMPTOM] 
  'treated_with',          // [DISEASE|CLINICAL_SYMPTOM] → TREATMENT
  'confers_resistance_to', // GENE → PATHOGEN (NEW in v2.0)
  'resistant_to',          // [SPECIES|PRODUCT] → [PATHOGEN|TREATMENT]
  
  // Risk and protective factors (NEW in v2.0)
  'increases_risk_of',     // [ENVIRONMENTAL_PARAM|MANAGEMENT_PRACTICE] → [DISEASE|CLINICAL_SYMPTOM]
  'reduces_risk_of',       // [MANAGEMENT_PRACTICE|TREATMENT] → [DISEASE|CLINICAL_SYMPTOM]
  
  // Genetic and physiological (v2.0)
  'expressed_in',          // GENE → [TISSUE|LIFE_STAGE] (NEW in v2.0)
  'affects_trait',         // [GENE|ENVIRONMENTAL_PARAM|MANAGEMENT_PRACTICE] → PHENOTYPIC_TRAIT
  'has_variant',           // GENE → GENE
  
  // Sampling and testing (NEW in v2.0)
  'sample_taken_from',     // SAMPLE → [LOCATION|SPECIES|TISSUE]
  'tested_with',           // SAMPLE → TEST_TYPE
  'has_test_result',       // SAMPLE → TEST_RESULT
  'measurement_of',        // MEASUREMENT → [PHENOTYPIC_TRAIT|ENVIRONMENTAL_PARAM]
  
  // Operational (NEW in v2.0)
  'applied_at',            // [TREATMENT|MANAGEMENT_PRACTICE] → [LOCATION|EVENT]
  'located_in',            // * → LOCATION
  'supplied_by',           // [PRODUCT|TREATMENT] → [SUPPLY_ENTITY|ORGANIZATION]
  'sold_to',               // PRODUCT → [ORGANIZATION|PERSON]
  'uses_protocol',         // [EVENT|ORGANIZATION|PERSON] → PROTOCOL
  'certified_by',          // [PRODUCT|ORGANIZATION|LOCATION] → CERTIFICATION
  'part_of',               // [LOCATION|PRODUCT|SAMPLE] → [LOCATION|PRODUCT]
];

function AnnotationWorkspace() {
  const { itemId } = useParams();
  const navigate = useNavigate();
  
  // State management
  const [currentItem, setCurrentItem] = useState(null);
  const [entities, setEntities] = useState([]);
  const [relations, setRelations] = useState([]);
  const [topics, setTopics] = useState([]);
  const [selectedEntities, setSelectedEntities] = useState([]);
  const [annotationMode, setAnnotationMode] = useState('entity'); // entity, relation, topic
  const [selectedEntityType, setSelectedEntityType] = useState('SPECIES');
  const [notes, setNotes] = useState('');
  const [confidence, setConfidence] = useState('high');
  const [showGuidelines, setShowGuidelines] = useState(false);
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const [showModifyDialog, setShowModifyDialog] = useState(false);
  const [showDraftDialog, setShowDraftDialog] = useState(false);
  const [loading, setLoading] = useState(false);
  
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
    connectionStatus,
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

  // Check for draft on item load
  useEffect(() => {
    if (currentItem && isDrafted) {
      setShowDraftDialog(true);
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
      console.log('🔄 Restored draft data');
    }
    setShowDraftDialog(false);
  };

  const handleDiscardDraft = () => {
    clearDraft();
    setShowDraftDialog(false);
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

  const loadNextItem = async () => {
    setLoading(true);
    try {
      const item = await getNextItem();
      if (item) {
        navigate(`/annotate/${item.id}`);
      } else {
        // No more items
        navigate('/triage');
      }
    } catch (error) {
      console.error('Failed to load next item:', error);
    } finally {
      setLoading(false);
    }
  };

  // Enhanced keyboard shortcuts
  useHotkeys('ctrl+enter', () => handleAccept(), { preventDefault: true });
  useHotkeys('ctrl+r', () => handleReject(), { preventDefault: true });
  useHotkeys('ctrl+m', () => handleModify(), { preventDefault: true });
  useHotkeys('ctrl+shift+s', () => handleSkip(), { preventDefault: true });
  useHotkeys('ctrl+s', () => handleSave(), { preventDefault: true });
  useHotkeys('ctrl+n', () => loadNextItem(), { preventDefault: true });
  useHotkeys('escape', () => setShowModifyDialog(false), { preventDefault: true });
  
  // Entity type shortcuts (v2.0 ontology)
  useHotkeys('1', () => setSelectedEntityType('SPECIES'));
  useHotkeys('2', () => setSelectedEntityType('PATHOGEN'));
  useHotkeys('3', () => setSelectedEntityType('DISEASE'));
  useHotkeys('4', () => setSelectedEntityType('CLINICAL_SYMPTOM'));
  useHotkeys('5', () => setSelectedEntityType('PHENOTYPIC_TRAIT'));
  useHotkeys('6', () => setSelectedEntityType('GENE'));
  useHotkeys('7', () => setSelectedEntityType('TREATMENT'));
  useHotkeys('8', () => setSelectedEntityType('LIFE_STAGE'));
  useHotkeys('9', () => setSelectedEntityType('MEASUREMENT'));
  useHotkeys('0', () => setSelectedEntityType('LOCATION'));
  
  // Mode switching
  useHotkeys('e', () => setAnnotationMode('entity'));
  useHotkeys('r', () => setAnnotationMode('relation'));
  useHotkeys('t', () => setAnnotationMode('topic'));
  
  // Help
  useHotkeys('?', () => setShowKeyboardHelp(true));
  useHotkeys('h', () => setShowKeyboardHelp(true));
  useHotkeys('f1', () => setShowGuidelines(true), { preventDefault: true });

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
  const handleAccept = async () => {
    if (!currentItem) return;
    
    const annotation = {
      item_id: currentItem.item_id || currentItem.id,
      candidate_id: currentItem.candidate_id || currentItem.item_id || currentItem.id,
      decision: 'accept',
      entities: entities,
      relations: relations,
      topics: topics,
      confidence: confidence === 'high' ? 0.9 : confidence === 'medium' ? 0.7 : 0.5,
      notes: notes,
      user_id: 1
    };
    
    try {
      console.log('📝 Submitting accept annotation:', annotation);
      
      // Use network recovery for critical operations
      const response = await callWithRecovery('/api/annotations/decide', {
        method: 'POST',
        data: annotation
      }, {
        type: 'submit_annotation',
        itemId: currentItem.id,
        critical: true
      });
      
      updateSessionStats('completed');
      
      // Clear draft on successful submission
      clearDraft();
      stopAutoSave();
      
      // Move to next item if available
      if (response.next_item) {
        console.log('➡️ Moving to next item:', response.next_item.doc_id + '/' + response.next_item.sent_id);
        setCurrentItem(response.next_item);
        setEntities(response.next_item.candidates?.entities || []);
        setRelations(response.next_item.candidates?.relations || []);
        setTopics(response.next_item.candidates?.topics || []);
        navigate(`/annotate/${response.next_item.item_id || response.next_item.id}`);
      } else {
        console.log('✅ No more items, returning to triage queue');
        navigate('/triage');
      }
    } catch (error) {
      console.error('Failed to submit annotation:', error);
      
      if (error.queued) {
        // Show user that operation was queued
        console.log('📤 Annotation queued for retry due to network issue');
      }
    }
  };

  const handleReject = async () => {
    if (!currentItem) return;
    
    const annotation = {
      item_id: currentItem.item_id || currentItem.id,
      candidate_id: currentItem.candidate_id || currentItem.item_id || currentItem.id,
      decision: 'reject',
      notes: notes,
      user_id: 1
    };
    
    try {
      console.log('❌ Submitting reject annotation:', annotation);
      const response = await submitAnnotation(annotation);
      updateSessionStats('completed');
      
      // Move to next item if available
      if (response.next_item) {
        console.log('➡️ Moving to next item:', response.next_item.doc_id + '/' + response.next_item.sent_id);
        setCurrentItem(response.next_item);
        setEntities(response.next_item.candidates?.entities || []);
        setRelations(response.next_item.candidates?.relations || []);
        setTopics(response.next_item.candidates?.topics || []);
        navigate(`/annotate/${response.next_item.item_id || response.next_item.id}`);
      } else {
        console.log('✅ No more items, returning to triage queue');
        navigate('/triage');
      }
    } catch (error) {
      console.error('Failed to reject annotation:', error);
    }
  };

  const handleModify = async () => {
    if (!currentItem) return;
    
    const annotation = {
      candidate_id: currentItem.candidate_id,
      decision: 'modified',
      final_annotation: {
        entities: entities,
        relations: relations,
        topics: topics,
        confidence: confidence,
        notes: notes
      },
      annotator: 'current-user@example.com',
    };
    
    try {
      await submitAnnotation(annotation);
      loadNextItem();
    } catch (error) {
      console.error('Failed to submit modification:', error);
    }
  };

  const handleSave = () => {
    // Save current progress without submitting
    console.log('Saving progress...');
  };

  const handleSkip = async () => {
    if (!currentItem) return;
    
    const annotation = {
      item_id: currentItem.item_id || currentItem.id,
      candidate_id: currentItem.candidate_id || currentItem.item_id || currentItem.id,
      decision: 'skip',
      notes: 'Skipped by user',
      user_id: 1
    };
    
    try {
      console.log('⏭️ Submitting skip annotation:', annotation);
      const response = await submitAnnotation(annotation);
      updateSessionStats('skipped');
      
      // Move to next item if available
      if (response.next_item) {
        console.log('➡️ Moving to next item:', response.next_item.doc_id + '/' + response.next_item.sent_id);
        setCurrentItem(response.next_item);
        setEntities(response.next_item.candidates?.entities || []);
        setRelations(response.next_item.candidates?.relations || []);
        setTopics(response.next_item.candidates?.topics || []);
        navigate(`/annotate/${response.next_item.item_id || response.next_item.id}`);
      } else {
        console.log('✅ No more items, returning to triage queue');
        navigate('/triage');
      }
    } catch (error) {
      console.error('Failed to skip item:', error);
    }
  };

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
      {/* Real-time collaboration alerts */}
      <SystemAlerts alerts={systemAlerts} onDismiss={dismissAlert} />
      <CollaborationConflicts 
        conflicts={getCollaborationConflicts()} 
        onResolve={(itemId) => {
          // Handle conflict resolution if needed
          console.log('Resolving conflict for item:', itemId);
        }}
      />

      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container alignItems="center" spacing={2}>
          <Grid item xs>
            <Typography variant="h6">
              Annotation Workspace
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Item: {currentItem.id} | Priority: {currentItem.priority_level}
            </Typography>
          </Grid>
          <Grid item>
            <Chip 
              label={`Score: ${currentItem.priority_score?.toFixed(2) || 'N/A'}`}
              color="primary"
              variant="outlined"
            />
          </Grid>
          <Grid item>
            <Box sx={{ textAlign: 'center', minWidth: 120 }}>
              <Typography variant="caption" display="block" color="text.secondary">
                Session Progress
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                ✅ {sessionStats.itemsCompleted} | ⏭️ {sessionStats.itemsSkipped}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {sessionStats.averageTimePerItem > 0 ? 
                  `Avg: ${Math.round(sessionStats.averageTimePerItem / 1000)}s` : 
                  'Starting...'}
              </Typography>
            </Box>
          </Grid>
          <Grid item>
            <ConnectionStatus 
              isConnected={isConnected} 
              connectionStatus={connectionStatus} 
            />
          </Grid>
          <Grid item>
            <CollaborationIndicator
              itemId={currentItem?.id}
              usersOnItem={getUsersOnItem(currentItem?.id)}
              currentUserId={localStorage.getItem('user_id') || 'anonymous'}
            />
          </Grid>
          <Grid item>
            <Tooltip title="Guidelines (F1)">
              <IconButton onClick={() => setShowGuidelines(true)}>
                <HelpIcon />
              </IconButton>
            </Tooltip>
          </Grid>
          <Grid item>
            <Tooltip title="Keyboard Shortcuts (?)">
              <IconButton onClick={() => setShowKeyboardHelp(true)}>
                <KeyboardIcon />
              </IconButton>
            </Tooltip>
          </Grid>
        </Grid>
      </Paper>

      <Grid container spacing={3} sx={{ flexGrow: 1 }}>
        {/* Main annotation area */}
        <Grid item xs={8}>
          <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Sentence display */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Sentence
              </Typography>
              <Paper 
                variant="outlined" 
                sx={{ p: 2, mb: 2, minHeight: 100, cursor: annotationMode === 'entity' ? 'text' : 'default' }}
                onMouseUp={handleTextSelection}
              >
                <Typography 
                  id="sentence-text"
                  variant="body1" 
                  sx={{ lineHeight: 1.8, userSelect: annotationMode === 'entity' ? 'text' : 'none' }}
                >
                  {currentItem.text || 'No text available'}
                </Typography>
              </Paper>
            </Box>

            {/* Mode selector */}
            <ButtonGroup sx={{ mb: 2 }}>
              <Button 
                variant={annotationMode === 'entity' ? 'contained' : 'outlined'}
                onClick={() => setAnnotationMode('entity')}
              >
                Entities
              </Button>
              <Button 
                variant={annotationMode === 'relation' ? 'contained' : 'outlined'}
                onClick={() => setAnnotationMode('relation')}
              >
                Relations
              </Button>
              <Button 
                variant={annotationMode === 'topic' ? 'contained' : 'outlined'}
                onClick={() => setAnnotationMode('topic')}
              >
                Topics
              </Button>
            </ButtonGroup>

            {/* Annotation components */}
            {annotationMode === 'entity' && (
              <EntityAnnotator
                entities={entities}
                setEntities={setEntities}
                entityTypes={ENTITY_TYPES}
                selectedType={selectedEntityType}
                setSelectedType={setSelectedEntityType}
              />
            )}

            {annotationMode === 'relation' && (
              <RelationAnnotator
                entities={entities}
                relations={relations}
                onRelationsChange={setRelations}
                relationTypes={RELATION_TYPES}
              />
            )}

            {annotationMode === 'topic' && (
              <TopicAnnotator
                topics={topics}
                setTopics={setTopics}
              />
            )}
          </Paper>
        </Grid>

        {/* Right sidebar */}
        <Grid item xs={4}>
          <Grid container direction="column" spacing={2}>
            {/* Candidates panel */}
            <Grid item>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    AI Suggestions
                  </Typography>
                  {currentItem.candidate_data?.entities?.length > 0 && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Entities
                      </Typography>
                      {currentItem.candidate_data.entities.map((entity, index) => (
                        <Chip
                          key={index}
                          label={`${entity.text} (${entity.label})`}
                          size="small"
                          sx={{ m: 0.5 }}
                          color="primary"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  )}
                  
                  {currentItem.rule_results?.entities?.length > 0 && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Rule-based
                      </Typography>
                      {currentItem.rule_results.entities.map((entity, index) => (
                        <Chip
                          key={index}
                          label={`${entity.text} (${entity.label})`}
                          size="small"
                          sx={{ m: 0.5 }}
                          color="secondary"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Annotation controls */}
            <Grid item>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Annotation
                  </Typography>
                  
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Confidence</InputLabel>
                    <Select
                      value={confidence}
                      onChange={(e) => setConfidence(e.target.value)}
                    >
                      <MenuItem value="high">High</MenuItem>
                      <MenuItem value="medium">Medium</MenuItem>
                      <MenuItem value="low">Low</MenuItem>
                    </Select>
                  </FormControl>

                  <TextField
                    label="Notes"
                    multiline
                    rows={3}
                    fullWidth
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Add notes about this annotation..."
                  />
                </CardContent>
              </Card>
            </Grid>

            {/* Action buttons */}
            <Grid item>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Actions
                  </Typography>
                  
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Button
                        fullWidth
                        variant="contained"
                        color="success"
                        startIcon={<AcceptIcon />}
                        onClick={handleAccept}
                        disabled={isLoading}
                      >
                        Accept
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        fullWidth
                        variant="contained"
                        color="warning"
                        startIcon={<EditIcon />}
                        onClick={handleModify}
                        disabled={isLoading}
                      >
                        Modify
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        fullWidth
                        variant="outlined"
                        color="error"
                        startIcon={<RejectIcon />}
                        onClick={handleReject}
                        disabled={isLoading}
                      >
                        Reject
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<SkipIcon />}
                        onClick={handleSkip}
                        disabled={isLoading}
                      >
                        Skip
                      </Button>
                    </Grid>
                  </Grid>

                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Keyboard shortcuts:
                    </Typography>
                    <Typography variant="caption" display="block">
                      Ctrl+Enter: Accept | Ctrl+R: Reject
                    </Typography>
                    <Typography variant="caption" display="block">
                      Ctrl+S: Save | Ctrl+N: Next
                    </Typography>
                    <Typography variant="caption" display="block">
                      1-0: Entity types | ?: Help
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Guidelines dialog */}
      <Dialog 
        open={showGuidelines} 
        onClose={() => setShowGuidelines(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Annotation Guidelines</DialogTitle>
        <DialogContent>
          <Typography variant="body1" paragraph>
            Quick reference for shrimp aquaculture annotation:
          </Typography>
          
          <Typography variant="h6" gutterBottom>Entity Types</Typography>
          {Object.entries(ENTITY_TYPES).map(([type, config]) => (
            <Box key={type} sx={{ mb: 1 }}>
              <Chip 
                label={type} 
                sx={{ backgroundColor: config.color, color: 'white', mr: 1 }}
                size="small"
              />
              <Typography variant="body2" component="span">
                {config.description}
              </Typography>
            </Box>
          ))}

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            Key Guidelines
          </Typography>
          <Typography variant="body2" component="div">
            <ul>
              <li>Select text precisely - include only the entity span</li>
              <li>Use canonical forms when possible (e.g., "Penaeus vannamei" not "white shrimp")</li>
              <li>Only annotate explicit relationships, not implied ones</li>
              <li>Mark uncertain annotations with low confidence</li>
              <li>Add notes for ambiguous cases</li>
            </ul>
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowGuidelines(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Keyboard Help Dialog */}
      <Dialog 
        open={showKeyboardHelp} 
        onClose={() => setShowKeyboardHelp(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Keyboard Shortcuts</DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>Actions</Typography>
              <Box sx={{ '& > div': { mb: 1 } }}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>Ctrl+Enter</strong></Typography>
                  <Typography variant="body2">Accept</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>Ctrl+R</strong></Typography>
                  <Typography variant="body2">Reject</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>Ctrl+M</strong></Typography>
                  <Typography variant="body2">Modify</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>Ctrl+Shift+S</strong></Typography>
                  <Typography variant="body2">Skip</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>Ctrl+N</strong></Typography>
                  <Typography variant="body2">Next Item</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>Ctrl+S</strong></Typography>
                  <Typography variant="body2">Save Progress</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>Esc</strong></Typography>
                  <Typography variant="body2">Close Dialog</Typography>
                </Box>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>Modes & Types</Typography>
              <Box sx={{ '& > div': { mb: 1 } }}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>E</strong></Typography>
                  <Typography variant="body2">Entity Mode</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>R</strong></Typography>
                  <Typography variant="body2">Relation Mode</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>T</strong></Typography>
                  <Typography variant="body2">Topic Mode</Typography>
                </Box>
              </Box>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Entity Types</Typography>
              <Box sx={{ '& > div': { mb: 1 } }}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>1</strong></Typography>
                  <Typography variant="body2">SPECIES</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>2</strong></Typography>
                  <Typography variant="body2">PATHOGEN</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>3</strong></Typography>
                  <Typography variant="body2">DISEASE</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>4</strong></Typography>
                  <Typography variant="body2">CLINICAL_SYMPTOM</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>5</strong></Typography>
                  <Typography variant="body2">PHENOTYPIC_TRAIT</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>6</strong></Typography>
                  <Typography variant="body2">GENE</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>7</strong></Typography>
                  <Typography variant="body2">TREATMENT</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>8</strong></Typography>
                  <Typography variant="body2">LIFE_STAGE</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>9</strong></Typography>
                  <Typography variant="body2">MEASUREMENT</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>0</strong></Typography>
                  <Typography variant="body2">LOCATION</Typography>
                </Box>
              </Box>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Help</Typography>
              <Box sx={{ '& > div': { mb: 1 } }}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>? or H</strong></Typography>
                  <Typography variant="body2">Show this help</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>F1</strong></Typography>
                  <Typography variant="body2">Show guidelines</Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowKeyboardHelp(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Draft Restoration Dialog */}
      <Dialog
        open={showDraftDialog}
        onClose={() => setShowDraftDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Draft Found</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            A previous draft was found for this annotation item. Would you like to restore your work?
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            This will restore your previously saved entities, relations, topics, and notes.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDiscardDraft} color="error">
            Discard Draft
          </Button>
          <Button onClick={handleRestoreDraft} variant="contained">
            Restore Draft
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default AnnotationWorkspace;
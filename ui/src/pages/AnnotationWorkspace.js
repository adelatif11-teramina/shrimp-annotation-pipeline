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
import {
  ConnectionStatus,
  CollaborationIndicator,
  SystemAlerts,
  CollaborationConflicts
} from '../components/RealTimeIndicators';

// Entity type configurations
const ENTITY_TYPES = {
  SPECIES: { color: '#FF6B6B', description: 'Organism names' },
  PATHOGEN: { color: '#FF8E53', description: 'Disease agents' },
  DISEASE: { color: '#FFD93D', description: 'Clinical conditions' },
  TREATMENT: { color: '#6BCF7F', description: 'Medications, procedures' },
  GENE: { color: '#9B59B6', description: 'Genetic markers' },
  TRAIT: { color: '#E91E63', description: 'Measurable characteristics' },
  LIFE_STAGE: { color: '#00BCD4', description: 'Development stages' },
  MEASUREMENT: { color: '#795548', description: 'Numeric values with units' },
  ENVIRONMENTAL_PARAM: { color: '#607D8B', description: 'Environmental factors' },
  LOCATION: { color: '#9E9E9E', description: 'Geographic locations' },
  MANAGEMENT_PRACTICE: { color: '#4D96FF', description: 'Farming practices' },
};

// Relation types
const RELATION_TYPES = [
  // Co-reference types (for duplicate entities)
  'same_as', 'refers_to', 'abbreviation_of', 'synonym_of', 'alias_of',
  // Domain-specific relation types
  'infected_by', 'causes', 'treated_with', 'resistant_to',
  'associated_with', 'measurement_of', 'located_in', 
  'affects_trait', 'has_variant'
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
  const [loading, setLoading] = useState(false);
  
  // Session analytics
  const [sessionStats, setSessionStats] = useState({
    itemsCompleted: 0,
    itemsSkipped: 0,
    startTime: Date.now(),
    totalTimeSpent: 0,
    averageTimePerItem: 0
  });
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
  
  // Entity type shortcuts
  useHotkeys('1', () => setSelectedEntityType('SPECIES'));
  useHotkeys('2', () => setSelectedEntityType('PATHOGEN'));
  useHotkeys('3', () => setSelectedEntityType('DISEASE'));
  useHotkeys('4', () => setSelectedEntityType('SYMPTOM'));
  useHotkeys('5', () => setSelectedEntityType('CHEMICAL'));
  useHotkeys('6', () => setSelectedEntityType('EQUIPMENT'));
  useHotkeys('7', () => setSelectedEntityType('LOCATION'));
  useHotkeys('8', () => setSelectedEntityType('MEASUREMENT'));
  useHotkeys('9', () => setSelectedEntityType('PROCESS'));
  useHotkeys('0', () => setSelectedEntityType('FEED'));
  
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
      candidate_id: currentItem.candidate_id,
      decision: 'accept',
      entities: entities,
      relations: relations,
      topics: topics,
      confidence: confidence === 'high' ? 0.9 : confidence === 'medium' ? 0.7 : 0.5,
      notes: notes
    };
    
    try {
      await submitAnnotation(annotation);
      updateSessionStats('completed');
      loadNextItem();
    } catch (error) {
      console.error('Failed to submit annotation:', error);
    }
  };

  const handleReject = async () => {
    if (!currentItem) return;
    
    const annotation = {
      candidate_id: currentItem.candidate_id,
      decision: 'rejected',
      annotator: 'current-user@example.com',
      notes: notes
    };
    
    try {
      await submitAnnotation(annotation);
      loadNextItem();
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
    
    try {
      updateSessionStats('skipped');
      loadNextItem();
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
                      1-4: Entity types | ?: Help
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
                  <Typography variant="body2">SYMPTOM</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>5</strong></Typography>
                  <Typography variant="body2">CHEMICAL</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2"><strong>6-0</strong></Typography>
                  <Typography variant="body2">More types...</Typography>
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
    </Box>
  );
}

export default AnnotationWorkspace;
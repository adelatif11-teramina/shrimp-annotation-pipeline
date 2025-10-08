import React, { useState } from 'react';
import {
  Box,
  Typography,
  Chip,
  Button,
  ButtonGroup,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Card,
  CardContent,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Edit as EditIcon,
  Add as AddIcon,
  Warning as WarningIcon,
  Link as LinkIcon,
} from '@mui/icons-material';

function EntityAnnotator({ entities, setEntities, entityTypes, selectedType, setSelectedType }) {
  const [editingEntity, setEditingEntity] = useState(null);
  const [newEntityText, setNewEntityText] = useState('');
  const [selectedEntitiesForMerge, setSelectedEntitiesForMerge] = useState([]);
  const [showMergeDialog, setShowMergeDialog] = useState(false);
  const [duplicateGroups, setDuplicateGroups] = useState([]);
  const [showDuplicateAlert, setShowDuplicateAlert] = useState(false);

  const handleDeleteEntity = (entityId) => {
    setEntities(prev => prev.filter(e => e.id !== entityId));
  };

  const handleEditEntity = (entity) => {
    setEditingEntity(entity);
  };

  const handleSaveEdit = (entityId, updates) => {
    setEntities(prev => prev.map(e => 
      e.id === entityId ? { ...e, ...updates } : e
    ));
    setEditingEntity(null);
  };

  const handleAddEntity = () => {
    if (!newEntityText.trim()) return;
    
    const newEntity = {
      id: `e${entities.length}`,
      text: newEntityText.trim(),
      label: selectedType,
      start: 0, // Would be calculated from text selection
      end: newEntityText.length,
      confidence: 0.9,
      source: 'manual'
    };
    
    setEntities(prev => [...prev, newEntity]);
    setNewEntityText('');
  };

  // Entity selection and merging handlers
  const handleEntitySelection = (entityId, isSelected) => {
    if (isSelected) {
      setSelectedEntitiesForMerge(prev => [...prev, entityId]);
    } else {
      setSelectedEntitiesForMerge(prev => prev.filter(id => id !== entityId));
    }
  };

  const handleMergeEntities = () => {
    if (selectedEntitiesForMerge.length < 2) return;
    
    const entitiesToMerge = entities.filter(e => selectedEntitiesForMerge.includes(e.id));
    const sortedEntities = entitiesToMerge.sort((a, b) => a.start - b.start);
    
    // Create merged entity
    const mergedEntity = {
      id: `merged_${Date.now()}_${Math.random().toString(36).substring(2)}`,
      text: sortedEntities.map(e => e.text).join(' / '), // Combined text
      label: sortedEntities[0].label, // Use first entity's label
      start: Math.min(...sortedEntities.map(e => e.start)),
      end: Math.max(...sortedEntities.map(e => e.end)),
      confidence: Math.max(...sortedEntities.map(e => e.confidence || 0.9)),
      source: 'merged',
      mergedFrom: sortedEntities.map(e => ({ id: e.id, text: e.text })) // Keep track of original entities
    };
    
    // Remove original entities and add merged entity
    setEntities(prev => [
      ...prev.filter(e => !selectedEntitiesForMerge.includes(e.id)),
      mergedEntity
    ]);
    
    // Clear selection
    setSelectedEntitiesForMerge([]);
    setShowMergeDialog(false);
  };

  // Duplicate detection functionality
  const detectDuplicates = (entities) => {
    const groups = [];
    const processed = new Set();
    
    entities.forEach((entity, i) => {
      if (processed.has(entity.id)) return;
      
      const duplicates = [entity];
      const entityTextLower = entity.text.toLowerCase().trim();
      
      entities.forEach((otherEntity, j) => {
        if (i !== j && !processed.has(otherEntity.id)) {
          const otherTextLower = otherEntity.text.toLowerCase().trim();
          
          // Exact match
          if (entityTextLower === otherTextLower) {
            duplicates.push(otherEntity);
            processed.add(otherEntity.id);
          }
          // Similar match (contains or very similar)
          else if (
            (entityTextLower.includes(otherTextLower) || otherTextLower.includes(entityTextLower)) &&
            Math.abs(entityTextLower.length - otherTextLower.length) <= 3
          ) {
            duplicates.push(otherEntity);
            processed.add(otherEntity.id);
          }
          // Abbreviation detection (e.g., "AHPND" vs "acute hepatopancreatic necrosis disease")
          else if (
            (entityTextLower.length <= 6 && otherTextLower.split(' ').map(w => w[0]).join('').toLowerCase() === entityTextLower) ||
            (otherTextLower.length <= 6 && entityTextLower.split(' ').map(w => w[0]).join('').toLowerCase() === otherTextLower)
          ) {
            duplicates.push(otherEntity);
            processed.add(otherEntity.id);
          }
        }
      });
      
      if (duplicates.length > 1) {
        groups.push(duplicates);
        duplicates.forEach(dup => processed.add(dup.id));
      } else {
        processed.add(entity.id);
      }
    });
    
    return groups;
  };

  // Auto-detect duplicates when entities change
  React.useEffect(() => {
    const groups = detectDuplicates(entities);
    setDuplicateGroups(groups);
    setShowDuplicateAlert(groups.length > 0);
  }, [entities]);

  const isDuplicate = (entityId) => {
    return duplicateGroups.some(group => group.some(entity => entity.id === entityId));
  };

  const getDuplicateGroup = (entityId) => {
    return duplicateGroups.find(group => group.some(entity => entity.id === entityId));
  };

  const handleSelectDuplicateGroup = (group) => {
    const entityIds = group.map(entity => entity.id);
    setSelectedEntitiesForMerge(entityIds);
  };

  const getEntityColor = (label) => {
    return entityTypes[label]?.color || '#9e9e9e';
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Entity Annotation
      </Typography>

      {/* Entity type selector */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Selected Entity Type
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, maxWidth: '100%' }}>
          {Object.entries(entityTypes).map(([type, config]) => (
            <Button
              key={type}
              variant={selectedType === type ? 'contained' : 'outlined'}
              onClick={() => setSelectedType(type)}
              size="small"
              sx={{
                backgroundColor: selectedType === type ? config.color : 'transparent',
                borderColor: config.color,
                color: selectedType === type ? 'white' : config.color,
                '&:hover': {
                  backgroundColor: config.color,
                  color: 'white',
                },
                minWidth: 'auto',
                fontSize: '0.75rem',
                py: 0.5,
                px: 1,
              }}
            >
              {type}
            </Button>
          ))}
        </Box>
        <Typography variant="caption" display="block" sx={{ mt: 1 }}>
          {entityTypes[selectedType]?.description}
        </Typography>
      </Box>

      {/* Manual entity addition */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Add Entity Manually
          </Typography>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={6}>
              <TextField
                size="small"
                fullWidth
                label="Entity Text"
                value={newEntityText}
                onChange={(e) => setNewEntityText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddEntity()}
              />
            </Grid>
            <Grid item xs={4}>
              <FormControl size="small" fullWidth>
                <InputLabel>Type</InputLabel>
                <Select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                >
                  {Object.keys(entityTypes).map(type => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={2}>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={handleAddEntity}
                disabled={!newEntityText.trim()}
              >
                Add
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Duplicate detection alert */}
      {showDuplicateAlert && duplicateGroups.length > 0 && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2 }}
          onClose={() => setShowDuplicateAlert(false)}
        >
          <Typography variant="subtitle2" gutterBottom>
            <WarningIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
            Potential Duplicates Detected ({duplicateGroups.length} groups)
          </Typography>
          {duplicateGroups.map((group, groupIndex) => (
            <Box key={groupIndex} sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2">
                Group {groupIndex + 1}:
              </Typography>
              {group.map((entity, entityIndex) => (
                <React.Fragment key={entity.id}>
                  <Chip
                    label={`"${entity.text}" (${entity.label})`}
                    size="small"
                    color="warning"
                    variant="outlined"
                  />
                  {entityIndex < group.length - 1 && <LinkIcon sx={{ fontSize: 16 }} />}
                </React.Fragment>
              ))}
              <Button
                size="small"
                variant="outlined"
                color="warning"
                onClick={() => handleSelectDuplicateGroup(group)}
              >
                Select for Merge
              </Button>
            </Box>
          ))}
          <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
            Consider merging these entities or creating co-reference relations (same_as, refers_to, etc.)
          </Typography>
        </Alert>
      )}

      {/* Entity merging controls */}
      {entities.length > 1 && (
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="subtitle2">
            Select entities to merge:
          </Typography>
          {selectedEntitiesForMerge.length >= 2 && (
            <Button
              variant="contained"
              color="primary"
              size="small"
              onClick={() => setShowMergeDialog(true)}
            >
              Merge {selectedEntitiesForMerge.length} Entities
            </Button>
          )}
          {selectedEntitiesForMerge.length > 0 && (
            <Button
              variant="outlined"
              size="small"
              onClick={() => setSelectedEntitiesForMerge([])}
            >
              Clear Selection
            </Button>
          )}
        </Box>
      )}

      {/* Current entities */}
      <Typography variant="subtitle2" gutterBottom>
        Current Entities ({entities.length})
      </Typography>
      
      {entities.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          No entities annotated yet. Select text above to add entities, or use the manual form.
        </Typography>
      ) : (
        <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
          {entities.map((entity, index) => (
            <EntityCard
              key={entity.id}
              entity={entity}
              index={index}
              entityTypes={entityTypes}
              editingEntity={editingEntity}
              onEdit={handleEditEntity}
              onSaveEdit={handleSaveEdit}
              onDelete={handleDeleteEntity}
              getEntityColor={getEntityColor}
              isSelected={selectedEntitiesForMerge.includes(entity.id)}
              onSelectionChange={handleEntitySelection}
              showSelection={entities.length > 1}
              isDuplicate={isDuplicate(entity.id)}
              duplicateGroup={getDuplicateGroup(entity.id)}
            />
          ))}
        </Box>
      )}

      {/* Instructions */}
      <Box sx={{ mt: 2, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary">
          <strong>Instructions:</strong>
          <br />
          • Select text in the sentence above to automatically create entities
          • Use keyboard shortcuts 1-4 to quickly switch entity types
          • Edit entities by clicking the edit icon
          • Ensure entity boundaries are precise (no extra spaces)
        </Typography>
      </Box>

      {/* Merge Dialog */}
      <Dialog open={showMergeDialog} onClose={() => setShowMergeDialog(false)}>
        <DialogTitle>Merge Entities</DialogTitle>
        <DialogContent>
          <Typography variant="body2" gutterBottom>
            You are about to merge the following entities:
          </Typography>
          {selectedEntitiesForMerge.map(entityId => {
            const entity = entities.find(e => e.id === entityId);
            return entity ? (
              <Box key={entityId} sx={{ my: 1, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                <Typography variant="body2">
                  <strong>"{entity.text}"</strong> ({entity.label})
                </Typography>
              </Box>
            ) : null;
          })}
          <Alert severity="info" sx={{ mt: 2 }}>
            The merged entity will combine the text and use the span range from the first to last entity.
            You can also create a "same_as" relation instead to preserve both entities.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowMergeDialog(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleMergeEntities}
            variant="contained"
            color="primary"
          >
            Merge Entities
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

function EntityCard({ entity, index, entityTypes, editingEntity, onEdit, onSaveEdit, onDelete, getEntityColor, isSelected, onSelectionChange, showSelection, isDuplicate, duplicateGroup }) {
  const [editText, setEditText] = useState(entity.text);
  const [editLabel, setEditLabel] = useState(entity.label);
  const [editConfidence, setEditConfidence] = useState(entity.confidence || 0.9);

  const isEditing = editingEntity?.id === entity.id;

  const handleSave = () => {
    onSaveEdit(entity.id, {
      text: editText,
      label: editLabel,
      confidence: editConfidence,
    });
  };

  const handleCancel = () => {
    setEditText(entity.text);
    setEditLabel(entity.label);
    setEditConfidence(entity.confidence || 0.9);
    onEdit(null);
  };

  return (
    <Card 
      variant="outlined" 
      sx={{ 
        mb: 1,
        ...(isDuplicate && {
          borderColor: 'warning.main',
          backgroundColor: 'warning.light',
          opacity: 0.9
        })
      }}
    >
      <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
        {isEditing ? (
          <Grid container spacing={1} alignItems="center">
            <Grid item xs={4}>
              <TextField
                size="small"
                fullWidth
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
              />
            </Grid>
            <Grid item xs={3}>
              <FormControl size="small" fullWidth>
                <Select
                  value={editLabel}
                  onChange={(e) => setEditLabel(e.target.value)}
                >
                  {Object.keys(entityTypes).map(type => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={2}>
              <TextField
                size="small"
                fullWidth
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                value={editConfidence}
                onChange={(e) => setEditConfidence(parseFloat(e.target.value))}
              />
            </Grid>
            <Grid item xs={3}>
              <Button size="small" onClick={handleSave} sx={{ mr: 1 }}>
                Save
              </Button>
              <Button size="small" onClick={handleCancel}>
                Cancel
              </Button>
            </Grid>
          </Grid>
        ) : (
          <Grid container alignItems="center" spacing={1}>
            {showSelection && (
              <Grid item xs={1}>
                <Checkbox
                  size="small"
                  checked={isSelected}
                  onChange={(e) => onSelectionChange(entity.id, e.target.checked)}
                />
              </Grid>
            )}
            <Grid item xs={showSelection ? 1 : 1}>
              <Typography variant="body2" color="text.secondary">
                {index + 1}
              </Typography>
            </Grid>
            <Grid item xs={showSelection ? 3 : 4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                {isDuplicate && (
                  <Tooltip 
                    title={`Potential duplicate with: ${duplicateGroup?.filter(e => e.id !== entity.id).map(e => e.text).join(', ')}`}
                  >
                    <WarningIcon sx={{ fontSize: 16, color: 'warning.main' }} />
                  </Tooltip>
                )}
                <Typography variant="body2" fontWeight="medium">
                  "{entity.text}"
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={3}>
              <Chip
                label={entity.label}
                size="small"
                sx={{
                  backgroundColor: getEntityColor(entity.label),
                  color: 'white',
                }}
              />
            </Grid>
            <Grid item xs={2}>
              <Typography variant="caption" color="text.secondary">
                {(entity.confidence * 100).toFixed(0)}%
              </Typography>
            </Grid>
            <Grid item xs={2}>
              <Tooltip title="Edit entity">
                <IconButton size="small" onClick={() => onEdit(entity)}>
                  <EditIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Delete entity">
                <IconButton size="small" onClick={() => onDelete(entity.id)} color="error">
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  );
}

export default EntityAnnotator;

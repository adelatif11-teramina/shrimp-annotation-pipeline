import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Grid,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Link as LinkIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';

const RELATION_TYPES = [
  'causes',
  'infected_by',
  'treated_with',
  'prevents',
  'affects',
  'located_in',
  'produced_by',
  'contains',
  'associated_with'
];

function RelationAnnotator({ 
  entities = [], 
  relations = [], 
  onRelationsChange,
  relationTypes = RELATION_TYPES,
  disabled = false 
}) {
  const [selectedEntities, setSelectedEntities] = useState([]);
  const [selectedRelationType, setSelectedRelationType] = useState('');
  const [currentRelations, setCurrentRelations] = useState(relations);

  useEffect(() => {
    setCurrentRelations(relations);
  }, [relations]);

  const handleEntityClick = (entity) => {
    if (disabled) return;
    
    if (selectedEntities.find(e => e.id === entity.id)) {
      // Deselect entity
      setSelectedEntities(prev => prev.filter(e => e.id !== entity.id));
    } else if (selectedEntities.length < 2) {
      // Select entity (max 2 for relation)
      setSelectedEntities(prev => [...prev, entity]);
    }
  };

  const handleAddRelation = () => {
    if (selectedEntities.length === 2 && selectedRelationType) {
      const newRelation = {
        id: Date.now(),
        head: selectedEntities[0],
        tail: selectedEntities[1],
        type: selectedRelationType,
        confidence: 1.0
      };

      const updatedRelations = [...currentRelations, newRelation];
      setCurrentRelations(updatedRelations);
      onRelationsChange(updatedRelations);

      // Clear selection
      setSelectedEntities([]);
      setSelectedRelationType('');
    }
  };

  const handleDeleteRelation = (relationId) => {
    const updatedRelations = currentRelations.filter(r => r.id !== relationId);
    setCurrentRelations(updatedRelations);
    onRelationsChange(updatedRelations);
  };

  const handleClearSelection = () => {
    setSelectedEntities([]);
    setSelectedRelationType('');
  };

  const getEntityColor = (entity) => {
    if (selectedEntities.find(e => e.id === entity.id)) {
      return 'primary';
    }
    return 'default';
  };

  const isRelationValid = () => {
    return selectedEntities.length === 2 && selectedRelationType;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Relation Annotation
      </Typography>

      {/* Instructions */}
      <Alert severity="info" sx={{ mb: 2 }}>
        Click two entities to select them, choose a relation type, then click "Add Relation" to create a relation.
      </Alert>

      {/* Entity Selection */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Available Entities
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
            {entities.map((entity) => (
              <Chip
                key={entity.id}
                label={`${entity.text} (${entity.label || entity.type || 'Unknown'})`}
                onClick={() => handleEntityClick(entity)}
                color={getEntityColor(entity)}
                variant={selectedEntities.find(e => e.id === entity.id) ? 'filled' : 'outlined'}
                clickable={!disabled}
                disabled={disabled}
              />
            ))}
          </Box>

          {/* Relation Creation Controls */}
          {selectedEntities.length > 0 && (
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={4}>
                <Typography variant="body2">
                  Selected: {selectedEntities.map(e => e.text).join(' + ')}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Relation Type</InputLabel>
                  <Select
                    value={selectedRelationType}
                    onChange={(e) => setSelectedRelationType(e.target.value)}
                    disabled={disabled}
                  >
                    {relationTypes.map((type) => (
                      <MenuItem key={type} value={type}>
                        {type}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<AddIcon />}
                    onClick={handleAddRelation}
                    disabled={!isRelationValid() || disabled}
                  >
                    Add Relation
                  </Button>
                  <IconButton
                    size="small"
                    onClick={handleClearSelection}
                    disabled={disabled}
                  >
                    <ClearIcon />
                  </IconButton>
                </Box>
              </Grid>
            </Grid>
          )}
        </CardContent>
      </Card>

      {/* Current Relations */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Current Relations ({currentRelations.length})
          </Typography>
          
          {currentRelations.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No relations annotated yet. Select two entities and a relation type to get started.
            </Typography>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {currentRelations.map((relation) => (
                <Box
                  key={relation.id}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 1,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    backgroundColor: 'background.paper'
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Chip
                      label={relation.head.text}
                      size="small"
                      color="primary"
                    />
                    <LinkIcon color="action" />
                    <Chip
                      label={relation.type}
                      size="small"
                      variant="outlined"
                    />
                    <LinkIcon color="action" />
                    <Chip
                      label={relation.tail.text}
                      size="small"
                      color="secondary"
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      {(relation.confidence * 100).toFixed(0)}%
                    </Typography>
                    <Tooltip title="Delete relation">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteRelation(relation.id)}
                        disabled={disabled}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}

export default RelationAnnotator;
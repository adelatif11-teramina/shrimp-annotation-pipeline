import React from 'react';
import {
  Paper,
  Box,
  Typography,
  ButtonGroup,
  Button,
} from '@mui/material';

import EntityAnnotator from '../EntityAnnotator';
import RelationAnnotator from '../RelationAnnotator';
import TopicAnnotator from '../TopicAnnotator';

function AnnotationMainPanel({
  currentItem,
  annotationMode,
  onChangeMode,
  onTextSelection,
  entities,
  setEntities,
  selectedEntityType,
  setSelectedEntityType,
  relations,
  setRelations,
  topics,
  setTopics,
  entityTypes,
  relationTypes,
}) {
  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Sentence
        </Typography>
        <Paper
          variant="outlined"
          sx={{
            p: 2,
            mb: 2,
            minHeight: 100,
            cursor: annotationMode === 'entity' ? 'text' : 'default',
          }}
          onMouseUp={onTextSelection}
        >
          <Typography
            id="sentence-text"
            variant="body1"
            sx={{
              lineHeight: 1.8,
              userSelect: annotationMode === 'entity' ? 'text' : 'none',
            }}
          >
            {currentItem.text || 'No text available'}
          </Typography>
        </Paper>
      </Box>

      <ButtonGroup sx={{ mb: 2 }}>
        <Button
          variant={annotationMode === 'entity' ? 'contained' : 'outlined'}
          onClick={() => onChangeMode('entity')}
        >
          Entities
        </Button>
        <Button
          variant={annotationMode === 'relation' ? 'contained' : 'outlined'}
          onClick={() => onChangeMode('relation')}
        >
          Relations
        </Button>
        <Button
          variant={annotationMode === 'topic' ? 'contained' : 'outlined'}
          onClick={() => onChangeMode('topic')}
        >
          Topics
        </Button>
      </ButtonGroup>

      {annotationMode === 'entity' && (
        <EntityAnnotator
          entities={entities}
          setEntities={setEntities}
          entityTypes={entityTypes}
          selectedType={selectedEntityType}
          setSelectedType={setSelectedEntityType}
        />
      )}

      {annotationMode === 'relation' && (
        <RelationAnnotator
          entities={entities}
          relations={relations}
          onRelationsChange={setRelations}
          relationTypes={relationTypes}
        />
      )}

      {annotationMode === 'topic' && (
        <TopicAnnotator topics={topics} setTopics={setTopics} />
      )}
    </Paper>
  );
}

export default AnnotationMainPanel;

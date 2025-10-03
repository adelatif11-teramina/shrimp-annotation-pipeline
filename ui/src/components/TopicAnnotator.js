import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Grid,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Topic as TopicIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
} from '@mui/icons-material';

const TOPIC_TYPES = [
  { id: 'T_DISEASE', label: 'Disease Management', color: '#f44336' },
  { id: 'T_TREATMENT', label: 'Treatment & Medicine', color: '#2196f3' },
  { id: 'T_PREVENTION', label: 'Prevention & Control', color: '#9c27b0' },
  { id: 'T_DIAGNOSIS', label: 'Diagnosis & Detection', color: '#673ab7' },
  { id: 'T_NUTRITION', label: 'Nutrition & Feed', color: '#4caf50' },
  { id: 'T_WATER_QUALITY', label: 'Water Quality', color: '#00bcd4' },
  { id: 'T_PRODUCTION', label: 'Production Systems', color: '#795548' },
  { id: 'T_HARVEST', label: 'Harvest & Processing', color: '#ff5722' },
  { id: 'T_ECONOMICS', label: 'Economics & Business', color: '#607d8b' },
  { id: 'T_GENERAL', label: 'General', color: '#9e9e9e' },
  { id: 'T_BREEDING', label: 'Breeding & Genetics', color: '#e91e63' },
  { id: 'T_ENVIRONMENT', label: 'Environmental Control', color: '#ff9800' },
  { id: 'T_RESEARCH', label: 'Research & Development', color: '#3f51b5' },
  { id: 'T_SUSTAINABILITY', label: 'Sustainability', color: '#8bc34a' }
];

function TopicAnnotator({ 
  topics = [], 
  onTopicsChange,
  setTopics,
  disabled = false,
  allowMultiple = true 
}) {
  // Support both prop names for backward compatibility
  const handleTopicsChange = onTopicsChange || setTopics || (() => {});
  const [currentTopics, setCurrentTopics] = useState(topics);
  const [selectedTopic, setSelectedTopic] = useState('');
  const [confidence, setConfidence] = useState(0.8);

  useEffect(() => {
    setCurrentTopics(topics);
  }, [topics]);

  const handleAddTopic = () => {
    if (!selectedTopic) return;

    const topicInfo = TOPIC_TYPES.find(t => t.id === selectedTopic);
    const newTopic = {
      id: Date.now(),
      topic_id: selectedTopic,
      label: topicInfo.label,
      confidence: confidence,
      color: topicInfo.color
    };

    // Check if topic already exists
    if (currentTopics.find(t => t.topic_id === selectedTopic)) {
      return; // Don't add duplicate topics
    }

    let updatedTopics;
    if (allowMultiple) {
      updatedTopics = [...currentTopics, newTopic];
    } else {
      updatedTopics = [newTopic]; // Replace existing topic
    }

    setCurrentTopics(updatedTopics);
    handleTopicsChange(updatedTopics);
    setSelectedTopic('');
  };

  const handleRemoveTopic = (topicId) => {
    const updatedTopics = currentTopics.filter(t => t.id !== topicId);
    setCurrentTopics(updatedTopics);
    handleTopicsChange(updatedTopics);
  };

  const getAvailableTopics = () => {
    if (allowMultiple) {
      return TOPIC_TYPES.filter(
        topic => !currentTopics.find(t => t.topic_id === topic.id)
      );
    }
    return TOPIC_TYPES;
  };

  const getTotalConfidence = () => {
    return currentTopics.reduce((sum, topic) => sum + topic.confidence, 0);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Topic Classification
      </Typography>

      {/* Instructions */}
      <Alert severity="info" sx={{ mb: 2 }}>
        {allowMultiple 
          ? 'Select one or more topics that best describe the content of this sentence.'
          : 'Select the primary topic that best describes the content of this sentence.'
        }
      </Alert>

      {/* Topic Selection */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Add Topic
          </Typography>
          
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Topic Type</InputLabel>
                <Select
                  value={selectedTopic}
                  onChange={(e) => setSelectedTopic(e.target.value)}
                  disabled={disabled}
                >
                  {getAvailableTopics().map((topic) => (
                    <MenuItem key={topic.id} value={topic.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box
                          sx={{
                            width: 12,
                            height: 12,
                            borderRadius: '50%',
                            backgroundColor: topic.color
                          }}
                        />
                        {topic.label}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <Typography variant="body2" gutterBottom>
                Confidence: {(confidence * 100).toFixed(0)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={confidence * 100}
                sx={{ height: 8, borderRadius: 4 }}
              />
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => setConfidence(0.6)}
                  disabled={disabled}
                >
                  Low
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => setConfidence(0.8)}
                  disabled={disabled}
                >
                  Med
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => setConfidence(0.95)}
                  disabled={disabled}
                >
                  High
                </Button>
              </Box>
            </Grid>
            
            <Grid item xs={12} sm={2}>
              <Button
                variant="contained"
                fullWidth
                startIcon={<AddIcon />}
                onClick={handleAddTopic}
                disabled={!selectedTopic || disabled}
              >
                Add
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Current Topics */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1">
              Current Topics ({currentTopics.length})
            </Typography>
            {allowMultiple && currentTopics.length > 0 && (
              <Typography variant="caption" color="text.secondary">
                Total Confidence: {(getTotalConfidence() * 100).toFixed(0)}%
              </Typography>
            )}
          </Box>
          
          {currentTopics.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No topics assigned yet. Select a topic type and add it to get started.
            </Typography>
          ) : (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {currentTopics.map((topic, index) => (
                <Chip
                  key={topic.id || index}
                  icon={<TopicIcon />}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: topic.color
                        }}
                      />
                      {topic.label}
                      <Typography variant="caption" component="span">
                        ({(topic.confidence * 100).toFixed(0)}%)
                      </Typography>
                    </Box>
                  }
                  onDelete={disabled ? undefined : () => handleRemoveTopic(topic.id)}
                  deleteIcon={<RemoveIcon />}
                  variant="filled"
                  sx={{
                    backgroundColor: `${topic.color}20`,
                    color: topic.color,
                    '& .MuiChip-deleteIcon': {
                      color: topic.color
                    }
                  }}
                />
              ))}
            </Box>
          )}

          {/* Topic Guidelines */}
          <Box sx={{ mt: 2, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
              Topic Guidelines:
            </Typography>
            <Typography variant="caption" color="text.secondary" component="ul" sx={{ pl: 2, m: 0 }}>
              <li>Disease Management: Diseases, pathogens, symptoms, diagnosis</li>
              <li>Treatment & Medicine: Antibiotics, vaccines, therapeutic interventions</li>
              <li>Nutrition & Feed: Feed formulation, nutritional requirements, feeding practices</li>
              <li>Water Quality: pH, dissolved oxygen, temperature, filtration</li>
              <li>Breeding & Genetics: Selective breeding, genetic improvement, reproduction</li>
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}

export default TopicAnnotator;
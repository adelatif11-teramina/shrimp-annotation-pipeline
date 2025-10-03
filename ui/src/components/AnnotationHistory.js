import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  IconButton,
  Tooltip,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Check as AcceptIcon,
  Close as RejectIcon,
  Edit as ModifyIcon,
  History as HistoryIcon,
  Visibility as ViewIcon,
  Undo as RestoreIcon,
} from '@mui/icons-material';

function AnnotationHistory({ 
  itemId,
  onRestoreVersion,
  compact = false 
}) {
  const [history, setHistory] = useState([]);
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    if (itemId) {
      fetchAnnotationHistory();
    }
  }, [itemId]);

  const fetchAnnotationHistory = async () => {
    // Mock annotation history data
    const mockHistory = [
      {
        id: 1,
        timestamp: new Date('2024-01-24T10:30:00'),
        action: 'created',
        annotator: 'alice@example.com',
        entities: [
          { text: 'Vibrio parahaemolyticus', type: 'PATHOGEN' },
          { text: 'AHPND', type: 'DISEASE' }
        ],
        relations: [
          { head: 'Vibrio parahaemolyticus', tail: 'AHPND', type: 'causes' }
        ],
        topics: ['T_DISEASE'],
        confidence: 0.95,
        notes: 'Initial annotation based on LLM suggestions'
      },
      {
        id: 2,
        timestamp: new Date('2024-01-24T11:15:00'),
        action: 'modified',
        annotator: 'bob@example.com',
        entities: [
          { text: 'Vibrio parahaemolyticus', type: 'PATHOGEN' },
          { text: 'AHPND', type: 'DISEASE' },
          { text: 'shrimp', type: 'SPECIES' }
        ],
        relations: [
          { head: 'Vibrio parahaemolyticus', tail: 'AHPND', type: 'causes' },
          { head: 'AHPND', tail: 'shrimp', type: 'affects' }
        ],
        topics: ['T_DISEASE', 'T_TREATMENT'],
        confidence: 0.92,
        notes: 'Added missing entity and relation'
      },
      {
        id: 3,
        timestamp: new Date('2024-01-24T14:20:00'),
        action: 'accepted',
        annotator: 'carol@example.com',
        entities: [
          { text: 'Vibrio parahaemolyticus', type: 'PATHOGEN' },
          { text: 'AHPND', type: 'DISEASE' },
          { text: 'shrimp', type: 'SPECIES' }
        ],
        relations: [
          { head: 'Vibrio parahaemolyticus', tail: 'AHPND', type: 'causes' },
          { head: 'AHPND', tail: 'shrimp', type: 'affects' }
        ],
        topics: ['T_DISEASE'],
        confidence: 0.98,
        notes: 'Final review and acceptance'
      }
    ];

    setHistory(mockHistory);
  };

  const getActionIcon = (action) => {
    switch (action) {
      case 'accepted': return <AcceptIcon color="success" />;
      case 'rejected': return <RejectIcon color="error" />;
      case 'modified': return <ModifyIcon color="warning" />;
      default: return <HistoryIcon color="action" />;
    }
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'accepted': return 'success';
      case 'rejected': return 'error';
      case 'modified': return 'warning';
      default: return 'default';
    }
  };

  const handleViewDetails = (version) => {
    setSelectedVersion(version);
    setShowDetails(true);
  };

  const handleRestore = (version) => {
    if (onRestoreVersion) {
      onRestoreVersion(version);
    }
    setShowDetails(false);
  };

  const formatTimestamp = (timestamp) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(timestamp);
  };

  if (compact) {
    return (
      <Box>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          Recent History ({history.length} versions)
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
          {history.slice(0, 3).map((version) => (
            <Chip
              key={version.id}
              size="small"
              label={`${version.action} by ${version.annotator.split('@')[0]}`}
              color={getActionColor(version.action)}
              variant="outlined"
              onClick={() => handleViewDetails(version)}
            />
          ))}
          {history.length > 3 && (
            <Chip
              size="small"
              label={`+${history.length - 3} more`}
              variant="outlined"
              color="default"
            />
          )}
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Annotation History
      </Typography>

      {history.length === 0 ? (
        <Card>
          <CardContent>
            <Typography variant="body2" color="text.secondary" textAlign="center">
              No annotation history available for this item.
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent sx={{ p: 0 }}>
            <List>
              {history.map((version, index) => (
                <React.Fragment key={version.id}>
                  <ListItem>
                    <ListItemIcon>
                      {getActionIcon(version.action)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" fontWeight="medium">
                            {version.action.charAt(0).toUpperCase() + version.action.slice(1)}
                          </Typography>
                          <Chip
                            label={version.annotator.split('@')[0]}
                            size="small"
                            variant="outlined"
                          />
                          <Typography variant="caption" color="text.secondary">
                            {formatTimestamp(version.timestamp)}
                          </Typography>
                        </Box>
                      }
                      secondary={
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" display="block">
                            {version.entities.length} entities, {version.relations.length} relations, {version.topics.length} topics
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Confidence: {(version.confidence * 100).toFixed(0)}%
                          </Typography>
                          {version.notes && (
                            <Typography variant="caption" display="block" sx={{ mt: 0.5, fontStyle: 'italic' }}>
                              "{version.notes}"
                            </Typography>
                          )}
                        </Box>
                      }
                    />
                    <Box>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleViewDetails(version)}
                        >
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                      {index > 0 && (
                        <Tooltip title="Restore This Version">
                          <IconButton
                            size="small"
                            onClick={() => handleRestore(version)}
                          >
                            <RestoreIcon />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Box>
                  </ListItem>
                  {index < history.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Version Details Dialog */}
      <Dialog open={showDetails} onClose={() => setShowDetails(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Version Details
          {selectedVersion && (
            <Typography variant="caption" display="block" color="text.secondary">
              {selectedVersion.action} by {selectedVersion.annotator} on {formatTimestamp(selectedVersion.timestamp)}
            </Typography>
          )}
        </DialogTitle>
        <DialogContent>
          {selectedVersion && (
            <Box>
              {/* Entities */}
              <Typography variant="subtitle2" gutterBottom>
                Entities ({selectedVersion.entities.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                {selectedVersion.entities.map((entity, idx) => (
                  <Chip
                    key={idx}
                    label={`${entity.text} (${entity.type})`}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>

              {/* Relations */}
              <Typography variant="subtitle2" gutterBottom>
                Relations ({selectedVersion.relations.length})
              </Typography>
              <Box sx={{ mb: 2 }}>
                {selectedVersion.relations.map((relation, idx) => (
                  <Typography key={idx} variant="body2" sx={{ mb: 0.5 }}>
                    {relation.head} → <strong>{relation.type}</strong> → {relation.tail}
                  </Typography>
                ))}
              </Box>

              {/* Topics */}
              <Typography variant="subtitle2" gutterBottom>
                Topics ({selectedVersion.topics.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                {selectedVersion.topics.map((topic, idx) => (
                  <Chip
                    key={idx}
                    label={topic}
                    size="small"
                    color="secondary"
                    variant="outlined"
                  />
                ))}
              </Box>

              {/* Notes */}
              {selectedVersion.notes && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Notes
                  </Typography>
                  <Typography variant="body2" sx={{ fontStyle: 'italic', p: 1, backgroundColor: 'grey.50', borderRadius: 1 }}>
                    {selectedVersion.notes}
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDetails(false)}>
            Close
          </Button>
          {selectedVersion && onRestoreVersion && (
            <Button
              variant="contained"
              startIcon={<RestoreIcon />}
              onClick={() => handleRestore(selectedVersion)}
            >
              Restore This Version
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default AnnotationHistory;
import React from 'react';
import {
  Paper,
  Grid,
  Typography,
  Chip,
  Box,
  Tooltip,
  IconButton,
} from '@mui/material';
import HelpIcon from '@mui/icons-material/Help';
import KeyboardIcon from '@mui/icons-material/Keyboard';

import {
  ConnectionStatus,
  CollaborationIndicator,
} from '../RealTimeIndicators';

function AnnotationHeader({
  currentItem,
  sessionStats,
  isConnected,
  connectionStatus,
  usersOnItem,
  currentUserId,
  onShowGuidelines,
  onShowKeyboardHelp,
}) {
  if (!currentItem) {
    return null;
  }

  const itemId = currentItem.item_id || currentItem.id;
  const priorityScore = currentItem.priority_score?.toFixed(2);

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Grid container alignItems="center" spacing={2}>
        <Grid item xs>
          <Typography variant="h6">Annotation Workspace</Typography>
          <Typography variant="body2" color="text.secondary">
            Item: {itemId} | Priority: {currentItem.priority_level || 'unknown'}
          </Typography>
        </Grid>
        <Grid item>
          <Chip
            label={`Score: ${priorityScore || 'N/A'}`}
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
              {sessionStats.averageTimePerItem > 0
                ? `Avg: ${Math.round(sessionStats.averageTimePerItem / 1000)}s`
                : 'Starting...'}
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
            itemId={itemId}
            usersOnItem={usersOnItem}
            currentUserId={currentUserId}
          />
        </Grid>
        <Grid item>
          <Tooltip title="Guidelines (F1)">
            <IconButton onClick={onShowGuidelines}>
              <HelpIcon />
            </IconButton>
          </Tooltip>
        </Grid>
        <Grid item>
          <Tooltip title="Keyboard Shortcuts (?)">
            <IconButton onClick={onShowKeyboardHelp}>
              <KeyboardIcon />
            </IconButton>
          </Tooltip>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default AnnotationHeader;

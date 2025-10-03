import React from 'react';
import {
  Box,
  Chip,
  Tooltip,
  Typography,
  Badge,
  Avatar,
  AvatarGroup,
  Alert,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Paper
} from '@mui/material';
import {
  People,
  Person,
  Warning,
  Info,
  Error,
  CheckCircle,
  Close,
  ExpandMore,
  ExpandLess,
  Wifi,
  WifiOff,
  Sync
} from '@mui/icons-material';
import { useState } from 'react';

const ConnectionStatus = ({ isConnected, connectionStatus }) => {
  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'success';
      case 'connecting': return 'warning';
      case 'reconnecting': return 'warning';
      case 'disconnected': return 'error';
      case 'error': return 'error';
      case 'failed': return 'error';
      default: return 'default';
    }
  };
  
  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return <Wifi />;
      case 'connecting': return <Sync className="rotating" />;
      case 'reconnecting': return <Sync className="rotating" />;
      default: return <WifiOff />;
    }
  };
  
  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'reconnecting': return 'Reconnecting...';
      case 'disconnected': return 'Disconnected';
      case 'error': return 'Connection Error';
      case 'failed': return 'Connection Failed';
      default: return 'Unknown';
    }
  };
  
  return (
    <Tooltip title={`Real-time collaboration: ${getStatusText()}`}>
      <Chip
        icon={getStatusIcon()}
        label={getStatusText()}
        color={getStatusColor()}
        size="small"
        variant="outlined"
      />
    </Tooltip>
  );
};

const UserPresence = ({ connectedUsers, currentUserId }) => {
  const [expanded, setExpanded] = useState(false);
  
  const otherUsers = connectedUsers.filter(user => user.user_id !== currentUserId);
  
  if (connectedUsers.length === 0) {
    return null;
  }
  
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Tooltip title={`${connectedUsers.length} users online`}>
        <Badge badgeContent={connectedUsers.length} color="primary">
          <People color="action" />
        </Badge>
      </Tooltip>
      
      {otherUsers.length > 0 && (
        <>
          <AvatarGroup max={3} sx={{ '& .MuiAvatar-root': { width: 24, height: 24, fontSize: '0.75rem' } }}>
            {otherUsers.slice(0, 3).map(user => (
              <Tooltip key={user.user_id} title={`${user.username} (${user.role})`}>
                <Avatar sx={{ bgcolor: getUserColor(user.user_id) }}>
                  {user.username[0].toUpperCase()}
                </Avatar>
              </Tooltip>
            ))}
          </AvatarGroup>
          
          {otherUsers.length > 3 && (
            <IconButton size="small" onClick={() => setExpanded(!expanded)}>
              {expanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          )}
        </>
      )}
      
      <Collapse in={expanded}>
        <Paper sx={{ position: 'absolute', top: '100%', right: 0, zIndex: 1000, mt: 1, minWidth: 250 }}>
          <List dense>
            {connectedUsers.map(user => (
              <ListItem key={user.user_id}>
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: getUserColor(user.user_id), width: 32, height: 32 }}>
                    {user.username[0].toUpperCase()}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={user.username}
                  secondary={`${user.role} • ${user.current_item ? `Working on ${user.current_item}` : 'Idle'}`}
                />
                {user.user_id === currentUserId && (
                  <Chip label="You" size="small" color="primary" />
                )}
              </ListItem>
            ))}
          </List>
        </Paper>
      </Collapse>
    </Box>
  );
};

const CollaborationIndicator = ({ itemId, usersOnItem, currentUserId }) => {
  if (!usersOnItem || usersOnItem.length === 0) {
    return null;
  }
  
  const otherUsers = usersOnItem.filter(user => user.user_id !== currentUserId);
  const isCurrentUserOnItem = usersOnItem.some(user => user.user_id === currentUserId);
  
  if (otherUsers.length === 0) {
    return null;
  }
  
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
      <Tooltip title={`${otherUsers.length} other user(s) working on this item`}>
        <Chip
          icon={<Person />}
          label={`+${otherUsers.length}`}
          size="small"
          color={otherUsers.length > 1 ? 'warning' : 'info'}
          variant="outlined"
        />
      </Tooltip>
      
      <AvatarGroup max={2} sx={{ '& .MuiAvatar-root': { width: 20, height: 20, fontSize: '0.6rem' } }}>
        {otherUsers.slice(0, 2).map(user => (
          <Tooltip key={user.user_id} title={user.username}>
            <Avatar sx={{ bgcolor: getUserColor(user.user_id) }}>
              {user.username[0].toUpperCase()}
            </Avatar>
          </Tooltip>
        ))}
      </AvatarGroup>
    </Box>
  );
};

const SystemAlerts = ({ alerts, onDismiss }) => {
  if (alerts.length === 0) {
    return null;
  }
  
  const getAlertSeverity = (level) => {
    switch (level) {
      case 'error':
      case 'critical':
        return 'error';
      case 'warning':
        return 'warning';
      case 'success':
        return 'success';
      default:
        return 'info';
    }
  };
  
  const getAlertIcon = (level) => {
    switch (level) {
      case 'error':
      case 'critical':
        return <Error />;
      case 'warning':
        return <Warning />;
      case 'success':
        return <CheckCircle />;
      default:
        return <Info />;
    }
  };
  
  return (
    <Box sx={{ position: 'fixed', top: 80, right: 16, zIndex: 1300, maxWidth: 400 }}>
      {alerts.map(alert => (
        <Alert
          key={alert.id}
          severity={getAlertSeverity(alert.level)}
          icon={getAlertIcon(alert.level)}
          action={
            <IconButton
              size="small"
              color="inherit"
              onClick={() => onDismiss(alert.id)}
            >
              <Close fontSize="small" />
            </IconButton>
          }
          sx={{ mb: 1 }}
        >
          <Box>
            <Typography variant="body2" component="div">
              {alert.message}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {alert.from_system ? 'System' : alert.from_user} • {new Date(alert.timestamp).toLocaleTimeString()}
            </Typography>
          </Box>
        </Alert>
      ))}
    </Box>
  );
};

const CollaborationConflicts = ({ conflicts, onResolve }) => {
  if (conflicts.length === 0) {
    return null;
  }
  
  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="h6" color="warning.main" gutterBottom>
        Collaboration Conflicts
      </Typography>
      {conflicts.map(conflict => (
        <Alert
          key={conflict.itemId}
          severity="warning"
          action={
            <IconButton
              size="small"
              color="inherit"
              onClick={() => onResolve(conflict.itemId)}
            >
              <Close fontSize="small" />
            </IconButton>
          }
          sx={{ mb: 1 }}
        >
          <Typography variant="body2">
            Multiple users working on item {conflict.itemId}:
          </Typography>
          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
            {conflict.users.map(user => (
              <Chip
                key={user.user_id}
                label={user.username}
                size="small"
                avatar={
                  <Avatar sx={{ bgcolor: getUserColor(user.user_id) }}>
                    {user.username[0].toUpperCase()}
                  </Avatar>
                }
              />
            ))}
          </Box>
        </Alert>
      ))}
    </Box>
  );
};

// Utility function to generate consistent colors for users
const getUserColor = (userId) => {
  const colors = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b'
  ];
  
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    hash = userId.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  return colors[Math.abs(hash) % colors.length];
};

// Export all components
export {
  ConnectionStatus,
  UserPresence,
  CollaborationIndicator,
  SystemAlerts,
  CollaborationConflicts,
  getUserColor
};
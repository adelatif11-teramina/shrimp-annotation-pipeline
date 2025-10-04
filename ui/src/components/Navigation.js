import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Divider,
  Badge,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Assignment as TriageIcon,
  Edit as AnnotateIcon,
  Folder as DocumentsIcon,
  Analytics as ResultsIcon,
  Assessment as QualityIcon,
  Settings as SettingsIcon,
  Science as ScienceIcon,
} from '@mui/icons-material';

const drawerWidth = 240;

const navigationItems = [
  { path: '/dashboard', label: 'Dashboard', icon: DashboardIcon },
  { path: '/triage', label: 'Triage Queue', icon: TriageIcon, badge: 'pending' },
  { path: '/annotate', label: 'Annotate', icon: AnnotateIcon },
  { path: '/documents', label: 'Documents', icon: DocumentsIcon },
  { path: '/results', label: 'Annotation Results', icon: ResultsIcon },
  { path: '/quality', label: 'Quality Control', icon: QualityIcon },
  { path: '/settings', label: 'Settings', icon: SettingsIcon },
];

function Navigation() {
  const navigate = useNavigate();
  const location = useLocation();

  // Mock data for badges - in real app would come from API
  const badgeData = {
    pending: 42, // pending triage items
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
    >
      <Toolbar>
        <ScienceIcon sx={{ mr: 1, color: 'primary.main' }} />
        <Typography variant="h6" noWrap component="div">
          Shrimp Annotator
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path || 
                          (item.path === '/annotate' && location.pathname.startsWith('/annotate'));
          
          return (
            <ListItem key={item.path} disablePadding>
              <ListItemButton
                selected={isActive}
                onClick={() => navigate(item.path)}
              >
                <ListItemIcon>
                  {item.badge && badgeData[item.badge] ? (
                    <Badge badgeContent={badgeData[item.badge]} color="error">
                      <Icon />
                    </Badge>
                  ) : (
                    <Icon />
                  )}
                </ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
    </Drawer>
  );
}

export default Navigation;
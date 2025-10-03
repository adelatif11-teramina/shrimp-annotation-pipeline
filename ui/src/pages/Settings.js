import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  FormControl,
  FormControlLabel,
  Switch,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  Alert,
  Tabs,
  Tab,
  Slider,
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RestoreIcon,
} from '@mui/icons-material';

function Settings() {
  const [activeTab, setActiveTab] = useState(0);
  const [settings, setSettings] = useState({
    // General Settings
    api_url: '',  // Use proxy from package.json to avoid CORS issues
    auto_save: true,
    notifications_enabled: true,
    theme: 'light',
    
    // Annotation Settings
    auto_accept_threshold: 0.95,
    enable_auto_accept: true,
    require_double_annotation: false,
    annotation_timeout: 30,
    
    // Quality Settings
    min_iaa_threshold: 0.80,
    quality_check_enabled: true,
    strict_validation: false,
    
    // Performance Settings
    cache_enabled: true,
    batch_size: 10,
    concurrent_workers: 2,
    
    // Model Retraining
    auto_retraining_enabled: true,
    retraining_threshold: 100,
    require_manual_approval: false,
  });
  
  const [saveStatus, setSaveStatus] = useState(null);

  const handleSettingChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSaveSettings = async () => {
    try {
      setSaveStatus('saving');
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Here you would normally make an API call to save settings
      // await updateSystemSettings(settings);
      
      setSaveStatus('success');
      setTimeout(() => setSaveStatus(null), 3000);
    } catch (error) {
      setSaveStatus('error');
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  const handleResetSettings = () => {
    setSettings({
      api_url: 'http://localhost:8000',
      auto_save: true,
      notifications_enabled: true,
      theme: 'light',
      auto_accept_threshold: 0.95,
      enable_auto_accept: true,
      require_double_annotation: false,
      annotation_timeout: 30,
      min_iaa_threshold: 0.80,
      quality_check_enabled: true,
      strict_validation: false,
      cache_enabled: true,
      batch_size: 10,
      concurrent_workers: 2,
      auto_retraining_enabled: true,
      retraining_threshold: 100,
      require_manual_approval: false,
    });
  };

  const TabPanel = ({ children, value, index }) => (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        System Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Configure annotation pipeline settings, quality controls, and system preferences
      </Typography>

      {/* Save Status Alert */}
      {saveStatus && (
        <Alert 
          severity={saveStatus === 'success' ? 'success' : saveStatus === 'error' ? 'error' : 'info'}
          sx={{ mb: 3 }}
        >
          {saveStatus === 'saving' && 'Saving settings...'}
          {saveStatus === 'success' && 'Settings saved successfully!'}
          {saveStatus === 'error' && 'Failed to save settings. Please try again.'}
        </Alert>
      )}

      {/* Settings Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
            <Tab label="General" />
            <Tab label="Annotation" />
            <Tab label="Quality Control" />
            <Tab label="Performance" />
            <Tab label="Model Retraining" />
          </Tabs>
        </Box>

        {/* General Settings */}
        <TabPanel value={activeTab} index={0}>
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="API Server URL"
                  value={settings.api_url}
                  onChange={(e) => handleSettingChange('api_url', e.target.value)}
                  helperText="Base URL for the annotation API server"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Theme</InputLabel>
                  <Select
                    value={settings.theme}
                    onChange={(e) => handleSettingChange('theme', e.target.value)}
                  >
                    <MenuItem value="light">Light</MenuItem>
                    <MenuItem value="dark">Dark</MenuItem>
                    <MenuItem value="auto">Auto</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.auto_save}
                      onChange={(e) => handleSettingChange('auto_save', e.target.checked)}
                    />
                  }
                  label="Auto-save annotations"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.notifications_enabled}
                      onChange={(e) => handleSettingChange('notifications_enabled', e.target.checked)}
                    />
                  }
                  label="Enable notifications"
                />
              </Grid>
            </Grid>
          </CardContent>
        </TabPanel>

        {/* Annotation Settings */}
        <TabPanel value={activeTab} index={1}>
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.enable_auto_accept}
                      onChange={(e) => handleSettingChange('enable_auto_accept', e.target.checked)}
                    />
                  }
                  label="Enable auto-accept for high-confidence annotations"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>
                  Auto-accept Threshold: {(settings.auto_accept_threshold * 100).toFixed(0)}%
                </Typography>
                <Slider
                  value={settings.auto_accept_threshold}
                  onChange={(e, value) => handleSettingChange('auto_accept_threshold', value)}
                  min={0.80}
                  max={0.99}
                  step={0.01}
                  marks
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Annotation Timeout (minutes)"
                  value={settings.annotation_timeout}
                  onChange={(e) => handleSettingChange('annotation_timeout', parseInt(e.target.value))}
                  helperText="Time limit for completing an annotation task"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.require_double_annotation}
                      onChange={(e) => handleSettingChange('require_double_annotation', e.target.checked)}
                    />
                  }
                  label="Require double annotation for critical items"
                />
              </Grid>
            </Grid>
          </CardContent>
        </TabPanel>

        {/* Quality Control Settings */}
        <TabPanel value={activeTab} index={2}>
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.quality_check_enabled}
                      onChange={(e) => handleSettingChange('quality_check_enabled', e.target.checked)}
                    />
                  }
                  label="Enable automatic quality checks"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>
                  Minimum IAA Threshold: {(settings.min_iaa_threshold * 100).toFixed(0)}%
                </Typography>
                <Slider
                  value={settings.min_iaa_threshold}
                  onChange={(e, value) => handleSettingChange('min_iaa_threshold', value)}
                  min={0.60}
                  max={0.95}
                  step={0.05}
                  marks
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.strict_validation}
                      onChange={(e) => handleSettingChange('strict_validation', e.target.checked)}
                    />
                  }
                  label="Enable strict validation rules"
                />
              </Grid>
            </Grid>
          </CardContent>
        </TabPanel>

        {/* Performance Settings */}
        <TabPanel value={activeTab} index={3}>
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.cache_enabled}
                      onChange={(e) => handleSettingChange('cache_enabled', e.target.checked)}
                    />
                  }
                  label="Enable LLM response caching"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Batch Processing Size"
                  value={settings.batch_size}
                  onChange={(e) => handleSettingChange('batch_size', parseInt(e.target.value))}
                  helperText="Number of items processed in each batch"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Concurrent Workers"
                  value={settings.concurrent_workers}
                  onChange={(e) => handleSettingChange('concurrent_workers', parseInt(e.target.value))}
                  helperText="Number of parallel processing workers"
                />
              </Grid>
            </Grid>
          </CardContent>
        </TabPanel>

        {/* Model Retraining Settings */}
        <TabPanel value={activeTab} index={4}>
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.auto_retraining_enabled}
                      onChange={(e) => handleSettingChange('auto_retraining_enabled', e.target.checked)}
                    />
                  }
                  label="Enable automatic model retraining"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Retraining Threshold"
                  value={settings.retraining_threshold}
                  onChange={(e) => handleSettingChange('retraining_threshold', parseInt(e.target.value))}
                  helperText="Minimum new annotations to trigger retraining"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.require_manual_approval}
                      onChange={(e) => handleSettingChange('require_manual_approval', e.target.checked)}
                    />
                  }
                  label="Require manual approval for retraining"
                />
              </Grid>
              <Grid item xs={12}>
                <Alert severity="info">
                  Automatic retraining helps keep models current with new annotation data. 
                  Configure thresholds carefully to balance model freshness with computational costs.
                </Alert>
              </Grid>
            </Grid>
          </CardContent>
        </TabPanel>

        <Divider />
        
        {/* Action Buttons */}
        <Box sx={{ p: 3, display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSaveSettings}
            disabled={saveStatus === 'saving'}
          >
            {saveStatus === 'saving' ? 'Saving...' : 'Save Settings'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<RestoreIcon />}
            onClick={handleResetSettings}
          >
            Reset to Defaults
          </Button>
        </Box>
      </Card>
    </Box>
  );
}

export default Settings;
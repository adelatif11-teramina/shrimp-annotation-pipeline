import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Pagination,
  Alert,
  Tabs,
  Tab,
  CircularProgress,
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Download as ExportIcon,
  Refresh as RefreshIcon,
  Analytics as StatsIcon,
  FilterList as FilterIcon,
  Check as AcceptIcon,
  Close as RejectIcon,
  SkipNext as SkipIcon,
  Edit as ModifyIcon,
} from '@mui/icons-material';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

function AnnotationResults() {
  const [annotations, setAnnotations] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedAnnotation, setSelectedAnnotation] = useState(null);
  const [showDetailDialog, setShowDetailDialog] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  
  // Filtering and pagination
  const [filters, setFilters] = useState({
    decision: 'all',
    doc_id: '',
    user_id: '',
    sort_by: 'created_at'
  });
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 20,
    total: 0
  });
  
  const { apiCall } = useAnnotationAPI();

  useEffect(() => {
    loadAnnotations();
    loadStatistics();
  }, [filters, pagination.page]);

  const loadAnnotations = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        ...filters,
        limit: pagination.limit,
        offset: (pagination.page - 1) * pagination.limit
      });
      
      // Remove empty filters
      Object.keys(filters).forEach(key => {
        if (!filters[key] || filters[key] === 'all') {
          params.delete(key);
        }
      });
      
      const response = await apiCall(`/api/annotations?${params}`);
      
      setAnnotations(response.annotations || []);
      setPagination(prev => ({
        ...prev,
        total: response.total || 0
      }));
    } catch (error) {
      console.error('Failed to load annotations:', error);
      setAnnotations([]);
    } finally {
      setLoading(false);
    }
  };

  const loadStatistics = async () => {
    try {
      const response = await apiCall('/api/annotations/statistics');
      setStatistics(response);
    } catch (error) {
      console.error('Failed to load statistics:', error);
    }
  };

  const handleViewDetail = async (annotationId) => {
    try {
      const response = await apiCall(`/api/annotations/${annotationId}`);
      setSelectedAnnotation(response.annotation);
      setShowDetailDialog(true);
    } catch (error) {
      console.error('Failed to load annotation detail:', error);
    }
  };

  const handleExport = async (format) => {
    try {
      const params = new URLSearchParams(filters);
      // Remove empty filters
      Object.keys(filters).forEach(key => {
        if (!filters[key] || filters[key] === 'all') {
          params.delete(key);
        }
      });
      params.set('format', format);
      
      window.open(`/api/annotations/export?${params}`, '_blank');
      setShowExportDialog(false);
    } catch (error) {
      console.error('Failed to export annotations:', error);
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
    setPagination(prev => ({ ...prev, page: 1 })); // Reset to first page
  };

  const handlePageChange = (event, newPage) => {
    setPagination(prev => ({ ...prev, page: newPage }));
  };

  const getDecisionIcon = (decision) => {
    switch (decision) {
      case 'accept': return <AcceptIcon color="success" />;
      case 'reject': return <RejectIcon color="error" />;
      case 'skip': return <SkipIcon color="warning" />;
      case 'modified': return <ModifyIcon color="info" />;
      default: return null;
    }
  };

  const getDecisionColor = (decision) => {
    switch (decision) {
      case 'accept': return 'success';
      case 'reject': return 'error';
      case 'skip': return 'warning';
      case 'modified': return 'info';
      default: return 'default';
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
  };

  if (loading && annotations.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading annotation results...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Annotation Results
      </Typography>
      
      <Typography variant="body1" color="text.secondary" gutterBottom>
        View and analyze completed annotations with detailed statistics and export capabilities.
      </Typography>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={selectedTab} onChange={(e, v) => setSelectedTab(v)}>
          <Tab label="Annotations" />
          <Tab label="Statistics" />
        </Tabs>
      </Box>

      {selectedTab === 0 && (
        <>
          {/* Filters and Controls */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={2}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Decision</InputLabel>
                    <Select
                      value={filters.decision}
                      onChange={(e) => handleFilterChange('decision', e.target.value)}
                    >
                      <MenuItem value="all">All Decisions</MenuItem>
                      <MenuItem value="accept">Accepted</MenuItem>
                      <MenuItem value="reject">Rejected</MenuItem>
                      <MenuItem value="skip">Skipped</MenuItem>
                      <MenuItem value="modified">Modified</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} sm={2}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Document ID"
                    value={filters.doc_id}
                    onChange={(e) => handleFilterChange('doc_id', e.target.value)}
                    placeholder="Filter by document..."
                  />
                </Grid>
                
                <Grid item xs={12} sm={2}>
                  <TextField
                    fullWidth
                    size="small"
                    label="User ID"
                    type="number"
                    value={filters.user_id}
                    onChange={(e) => handleFilterChange('user_id', e.target.value)}
                    placeholder="Filter by user..."
                  />
                </Grid>
                
                <Grid item xs={12} sm={2}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Sort By</InputLabel>
                    <Select
                      value={filters.sort_by}
                      onChange={(e) => handleFilterChange('sort_by', e.target.value)}
                    >
                      <MenuItem value="created_at">Date Created</MenuItem>
                      <MenuItem value="priority_score">Priority Score</MenuItem>
                      <MenuItem value="confidence">Confidence</MenuItem>
                      <MenuItem value="time_spent">Time Spent</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} sm={2}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={loadAnnotations}
                    disabled={loading}
                  >
                    Refresh
                  </Button>
                </Grid>
                
                <Grid item xs={12} sm={2}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<ExportIcon />}
                    onClick={() => setShowExportDialog(true)}
                  >
                    Export
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Results Summary */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Results Overview
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={3}>
                  <Typography variant="h4" color="primary">
                    {pagination.total}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Results
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="h4" color="success.main">
                    {annotations.filter(a => a.decision === 'accept').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Accepted
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="h4" color="error.main">
                    {annotations.filter(a => a.decision === 'reject').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Rejected
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="h4" color="warning.main">
                    {annotations.filter(a => a.decision === 'skip').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Skipped
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Annotations Table */}
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Decision</TableCell>
                  <TableCell>Document</TableCell>
                  <TableCell>Sentence</TableCell>
                  <TableCell>Entities</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Time Spent</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {annotations.map((annotation) => (
                  <TableRow key={annotation.id}>
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        #{annotation.id}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getDecisionIcon(annotation.decision)}
                        label={annotation.decision}
                        color={getDecisionColor(annotation.decision)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ maxWidth: 150 }}>
                        {annotation.document_title || annotation.doc_id || 'Unknown'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {annotation.doc_id} / {annotation.sent_id}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ maxWidth: 200 }}>
                        {annotation.sentence_preview || 'No text available'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {annotation.entity_count || 0} entities
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {annotation.relation_count || 0} relations
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <LinearProgress
                        variant="determinate"
                        value={(annotation.confidence || 0) * 100}
                        sx={{ width: 60, mr: 1 }}
                      />
                      <Typography variant="caption">
                        {((annotation.confidence || 0) * 100).toFixed(0)}%
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {annotation.time_spent_formatted || 'Unknown'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatDate(annotation.created_at)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleViewDetail(annotation.id)}
                        >
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Pagination */}
          {pagination.total > pagination.limit && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Pagination
                count={Math.ceil(pagination.total / pagination.limit)}
                page={pagination.page}
                onChange={handlePageChange}
                color="primary"
              />
            </Box>
          )}

          {annotations.length === 0 && !loading && (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" color="text.secondary">
                No annotations found
              </Typography>
              <Typography variant="body2" color="text.secondary" mt={1}>
                Try adjusting your filters or perform some annotations first.
              </Typography>
            </Box>
          )}
        </>
      )}

      {selectedTab === 1 && (
        <Box>
          {statistics ? (
            <Grid container spacing={3}>
              {/* Summary Statistics */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Summary Statistics
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6} md={2}>
                        <Typography variant="h4" color="primary">
                          {statistics.summary.total_annotations}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Annotations
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <Typography variant="h4" color="success.main">
                          {statistics.summary.acceptance_rate}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Acceptance Rate
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <Typography variant="h4" color="info.main">
                          {statistics.summary.avg_time_per_annotation}s
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Avg Time/Annotation
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <Typography variant="h4" color="success.main">
                          {statistics.summary.accepted}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Accepted
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <Typography variant="h4" color="error.main">
                          {statistics.summary.rejected}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Rejected
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <Typography variant="h4" color="warning.main">
                          {statistics.summary.skipped}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Skipped
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Entity Statistics */}
              {statistics.entity_stats.length > 0 && (
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Entity Types
                      </Typography>
                      {statistics.entity_stats.slice(0, 10).map((entity, index) => (
                        <Box key={entity.type} sx={{ mb: 1 }}>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">{entity.type}</Typography>
                            <Typography variant="body2" fontWeight="medium">
                              {entity.count}
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={(entity.count / statistics.entity_stats[0].count) * 100}
                            sx={{ height: 4, mt: 0.5 }}
                          />
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Document Statistics */}
              {statistics.by_document.length > 0 && (
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        By Document
                      </Typography>
                      {statistics.by_document.slice(0, 5).map((doc) => (
                        <Box key={doc.doc_id} sx={{ mb: 2 }}>
                          <Typography variant="body2" fontWeight="medium">
                            {doc.doc_title}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {doc.total} annotations • {doc.acceptance_rate}% accepted
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={doc.acceptance_rate}
                            sx={{ height: 4, mt: 0.5 }}
                          />
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          ) : (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <CircularProgress />
              <Typography variant="h6" sx={{ mt: 2 }}>
                Loading statistics...
              </Typography>
            </Box>
          )}
        </Box>
      )}

      {/* Detail Dialog */}
      <Dialog
        open={showDetailDialog}
        onClose={() => setShowDetailDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Annotation Details
          {selectedAnnotation && (
            <Chip
              icon={getDecisionIcon(selectedAnnotation.decision)}
              label={selectedAnnotation.decision}
              color={getDecisionColor(selectedAnnotation.decision)}
              size="small"
              sx={{ ml: 2 }}
            />
          )}
        </DialogTitle>
        <DialogContent>
          {selectedAnnotation && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Sentence
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                  <Typography variant="body1">
                    {selectedAnnotation.sentence_text || 'No text available'}
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Basic Information
                </Typography>
                <Typography variant="body2">
                  <strong>ID:</strong> #{selectedAnnotation.id}
                </Typography>
                <Typography variant="body2">
                  <strong>Document:</strong> {selectedAnnotation.document_title || selectedAnnotation.doc_id}
                </Typography>
                <Typography variant="body2">
                  <strong>User ID:</strong> {selectedAnnotation.user_id}
                </Typography>
                <Typography variant="body2">
                  <strong>Confidence:</strong> {((selectedAnnotation.confidence || 0) * 100).toFixed(0)}%
                </Typography>
                <Typography variant="body2">
                  <strong>Time Spent:</strong> {selectedAnnotation.time_spent_formatted}
                </Typography>
                <Typography variant="body2">
                  <strong>Created:</strong> {selectedAnnotation.created_at_formatted}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Annotation Content
                </Typography>
                <Typography variant="body2">
                  <strong>Entities:</strong> {selectedAnnotation.entity_count}
                </Typography>
                <Typography variant="body2">
                  <strong>Relations:</strong> {selectedAnnotation.relation_count}
                </Typography>
                <Typography variant="body2">
                  <strong>Topics:</strong> {selectedAnnotation.topic_count}
                </Typography>
                {selectedAnnotation.notes && (
                  <Typography variant="body2">
                    <strong>Notes:</strong> {selectedAnnotation.notes}
                  </Typography>
                )}
              </Grid>

              {/* Show entities if available */}
              {selectedAnnotation.entities && selectedAnnotation.entities.length > 0 && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Entities
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {selectedAnnotation.entities.map((entity, index) => (
                      <Chip
                        key={index}
                        label={`${entity.text} (${entity.label})`}
                        size="small"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDetailDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Export Dialog */}
      <Dialog open={showExportDialog} onClose={() => setShowExportDialog(false)}>
        <DialogTitle>Export Annotations</DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            Choose the export format for your annotation data:
          </Typography>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={4}>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => handleExport('json')}
              >
                JSON
              </Button>
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                Complete data structure
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => handleExport('csv')}
              >
                CSV
              </Button>
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                Spreadsheet format
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => handleExport('scibert')}
              >
                SciBERT
              </Button>
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                Training format
              </Typography>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowExportDialog(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default AnnotationResults;
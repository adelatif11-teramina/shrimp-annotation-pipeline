import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  LinearProgress,
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
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Visibility as ViewIcon,
  Schedule as PriorityIcon,
  Person as AssignIcon,
} from '@mui/icons-material';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

function TriageQueue() {
  const [triageItems, setTriageItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('priority');
  
  const { getTriageQueue, getTriageStatistics } = useAnnotationAPI();

  useEffect(() => {
    fetchTriageData();
  }, [filter, sortBy]);

  const fetchTriageData = async () => {
    try {
      setLoading(true);
      const filters = {
        status: filter !== 'all' ? filter : undefined,
        sort_by: sortBy,
        limit: 50
      };
      
      const response = await getTriageQueue(filters);
      // Handle both array and object response formats
      const items = response?.items || response || [];
      setTriageItems(Array.isArray(items) ? items : []);
    } catch (error) {
      console.error('Failed to fetch triage data:', error);
      setTriageItems([]);
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (score) => {
    if (score >= 0.8) return 'error';
    if (score >= 0.6) return 'warning';
    if (score >= 0.4) return 'info';
    return 'default';
  };

  const getPriorityLabel = (score) => {
    if (score >= 0.8) return 'Critical';
    if (score >= 0.6) return 'High';
    if (score >= 0.4) return 'Medium';
    return 'Low';
  };

  const handleStartAnnotation = (itemId) => {
    // Navigate to annotation workspace
    window.location.href = `/annotate/${itemId}`;
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Triage Queue
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Annotation Triage Queue
      </Typography>
      
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Prioritized items awaiting annotation. Items are ranked by confidence, novelty, and impact.
      </Typography>

      {/* Filters and Controls */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Filter by Status</InputLabel>
            <Select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            >
              <MenuItem value="all">All Items</MenuItem>
              <MenuItem value="pending">Pending</MenuItem>
              <MenuItem value="in_review">In Review</MenuItem>
              <MenuItem value="assigned">Assigned</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="priority">Priority Score</MenuItem>
              <MenuItem value="confidence">Confidence</MenuItem>
              <MenuItem value="novelty">Novelty</MenuItem>
              <MenuItem value="created_at">Date Created</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Button 
            variant="outlined" 
            onClick={fetchTriageData}
            fullWidth
          >
            Refresh Queue
          </Button>
        </Grid>
      </Grid>

      {/* Queue Statistics */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Queue Overview
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={3}>
              <Typography variant="h4" color="primary">
                {triageItems.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Items
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="h4" color="error">
                {triageItems.filter(item => item.priority_score >= 0.8).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Critical Priority
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="h4" color="warning.main">
                {triageItems.filter(item => item.priority_score >= 0.6 && item.priority_score < 0.8).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                High Priority
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="h4" color="success.main">
                {triageItems.filter(item => item.status === 'pending').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Available
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Triage Queue Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Document</TableCell>
              <TableCell>Sentence</TableCell>
              <TableCell>Priority</TableCell>
              <TableCell>Confidence</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Assigned To</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {triageItems.map((item, index) => (
              <TableRow key={item.id || item.item_id || index}>
                <TableCell>
                  <Typography variant="body2" fontWeight="medium">
                    {item.doc_id || item.document_title || 'Unknown'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ maxWidth: 300 }}>
                    {(item.text || item.sentence_text || 'No text available').substring(0, 80)}...
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    icon={<PriorityIcon />}
                    label={getPriorityLabel(item.priority_score || 0)}
                    color={getPriorityColor(item.priority_score || 0)}
                    size="small"
                  />
                  <Typography variant="caption" display="block" mt={0.5}>
                    {(item.priority_score || 0).toFixed(2)}
                  </Typography>
                </TableCell>
                <TableCell>
                  <LinearProgress
                    variant="determinate"
                    value={(item.confidence || item.candidate_data?.confidence || 0.5) * 100}
                    sx={{ width: 80, mr: 1 }}
                  />
                  <Typography variant="caption">
                    {((item.confidence || item.candidate_data?.confidence || 0.5) * 100).toFixed(0)}%
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={item.status || 'pending'}
                    size="small"
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {item.assigned_to || 'Unassigned'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box>
                    <Tooltip title={item.status === 'in_review' ? 'Already in review' : 'Start Annotation'}>
                      <span>
                        <IconButton
                          color="primary"
                          onClick={() => handleStartAnnotation(item.id || item.item_id)}
                          disabled={item.status === 'in_review'}
                        >
                          <StartIcon />
                        </IconButton>
                      </span>
                    </Tooltip>
                    <Tooltip title="View Details">
                      <IconButton color="default">
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Assign to Me">
                      <IconButton color="secondary">
                        <AssignIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {triageItems.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h6" color="text.secondary">
            No items in triage queue
          </Typography>
          <Typography variant="body2" color="text.secondary" mt={1}>
            All available items have been processed or no new documents have been ingested.
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default TriageQueue;
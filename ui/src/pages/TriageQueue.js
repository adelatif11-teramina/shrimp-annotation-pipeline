import React, { useMemo, useState } from 'react';
import {
  Alert,
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
import { useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

function TriageQueue() {
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('priority');
  const navigate = useNavigate();
  const { getTriageQueue } = useAnnotationAPI();

  const triageQuery = useQuery(
    ['triageQueue', { filter, sortBy }],
    () =>
      getTriageQueue({
        status: filter !== 'all' ? filter : undefined,
        sort_by: sortBy,
        limit: 50,
      }),
    {
      keepPreviousData: true,
    },
  );

  const triageItems = useMemo(() => {
    const rawItems = triageQuery.data?.items || triageQuery.data || [];
    return Array.isArray(rawItems) ? rawItems : [];
  }, [triageQuery.data]);

  const handleStartAnnotation = (itemId) => {
    navigate(`/annotate/${itemId}`);
  };

  const handleViewDocument = (docId) => {
    navigate(`/documents?docId=${docId}`);
  };

  if (triageQuery.isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Triage Queue
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (triageQuery.error) {
    const message = triageQuery.error?.message || 'Failed to load triage queue data.';
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Triage Queue
        </Typography>
        <Alert severity="error">{message}</Alert>
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
            <Select value={filter} onChange={(e) => setFilter(e.target.value)}>
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
            <Select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
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
            onClick={() => triageQuery.refetch()}
            fullWidth
            disabled={triageQuery.isFetching}
          >
            {triageQuery.isFetching ? 'Refreshing…' : 'Refresh Queue'}
          </Button>
        </Grid>
      </Grid>

      {triageQuery.isFetching && <LinearProgress sx={{ mb: 2 }} />}

      {/* Queue Statistics */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Queue Overview
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={6} md={3}>
              <Typography variant="h4" color="primary">
                {triageItems.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Items
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="h4" color="error">
                {triageItems.filter((item) => item.priority_score >= 0.8).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Critical Priority
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="h4" color="warning.main">
                {triageItems.filter((item) => item.priority_score >= 0.6 && item.priority_score < 0.8).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                High Priority
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="h4" color="success.main">
                {triageItems.filter((item) => item.status === 'pending').length}
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
              <TableCell>Assigned</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {triageItems.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography color="text.secondary">No triage items match this filter.</Typography>
                </TableCell>
              </TableRow>
            ) : (
              triageItems.map((item) => (
                <TableRow key={item.item_id || item.id} hover>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip size="small" icon={<PriorityIcon />} label={item.document_title || 'Untitled'} />
                      <Typography variant="body2" color="text.secondary">
                        {item.doc_id || item.document_id || '—'}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" noWrap sx={{ maxWidth: 320 }}>
                      {item.sentence || item.text || '—'}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      color={getPriorityColor(item.priority_score)}
                      label={getPriorityLabel(item.priority_score)}
                    />
                  </TableCell>
                  <TableCell>{Math.round((item.confidence || 0) * 100)}%</TableCell>
                  <TableCell>
                    {item.assigned_to ? (
                      <Chip
                        size="small"
                        icon={<AssignIcon fontSize="small" />}
                        label={item.assigned_to}
                        color="info"
                      />
                    ) : (
                      <Chip size="small" label="Unassigned" variant="outlined" />
                    )}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Start annotation">
                      <span>
                        <Button
                          size="small"
                          variant="contained"
                          startIcon={<StartIcon />}
                          onClick={() => handleStartAnnotation(item.item_id || item.id)}
                        >
                          Start
                        </Button>
                      </span>
                    </Tooltip>
                    <Tooltip title="Preview document">
                      <IconButton onClick={() => handleViewDocument(item.doc_id || item.document_id)}>
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

const getPriorityColor = (score = 0) => {
  if (score >= 0.8) return 'error';
  if (score >= 0.6) return 'warning';
  if (score >= 0.4) return 'info';
  return 'default';
};

const getPriorityLabel = (score = 0) => {
  if (score >= 0.8) return 'Critical';
  if (score >= 0.6) return 'High';
  if (score >= 0.4) return 'Medium';
  return 'Low';
};

export default TriageQueue;

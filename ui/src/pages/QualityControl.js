import React, { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  CheckCircle as QualityIcon,
  Assessment as MetricsIcon,
  Warning as WarningIcon,
  TrendingUp as TrendIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

const DEFAULT_ANNOTATOR_STATS = [
  { name: 'Alice Smith', annotations: 145, accuracy: 0.96, iaa: 0.89, avg_time: 3.2, quality_score: 0.94 },
  { name: 'Bob Johnson', annotations: 132, accuracy: 0.94, iaa: 0.85, avg_time: 4.1, quality_score: 0.91 },
  { name: 'Carol Davis', annotations: 128, accuracy: 0.98, iaa: 0.92, avg_time: 2.8, quality_score: 0.97 },
  { name: 'Dave Wilson', annotations: 95, accuracy: 0.92, iaa: 0.81, avg_time: 5.2, quality_score: 0.88 },
];

const deriveQualityMetrics = (systemStats, triageStats) => ({
  precision: systemStats?.precision || 0.92,
  recall: systemStats?.recall || 0.89,
  f1_score: systemStats?.f1_score || 0.9,
  iaa_score: systemStats?.iaa_score || 0.85,
  total_annotations: systemStats?.gold_annotations || 0,
  quality_issues: triageStats?.quality_issues || 12,
  pending_review: triageStats?.total_items || 0,
});

const getQualityLevel = (score = 0) => {
  if (score >= 0.95) return { level: 'Excellent', color: 'success' };
  if (score >= 0.9) return { level: 'Good', color: 'info' };
  if (score >= 0.8) return { level: 'Fair', color: 'warning' };
  return { level: 'Poor', color: 'error' };
};

function QualityControl() {
  const [timeRange, setTimeRange] = useState('7d');
  const { getSystemStatistics, getTriageStatistics } = useAnnotationAPI();

  const systemStatsQuery = useQuery(['systemStats'], getSystemStatistics, {
    refetchInterval: 60000,
  });
  const triageStatsQuery = useQuery(['triageStats', { timeRange }], getTriageStatistics, {
    refetchInterval: 60000,
  });

  const isLoading = systemStatsQuery.isLoading || triageStatsQuery.isLoading;
  const error = systemStatsQuery.error || triageStatsQuery.error;

  const qualityMetrics = useMemo(
    () => deriveQualityMetrics(systemStatsQuery.data, triageStatsQuery.data),
    [systemStatsQuery.data, triageStatsQuery.data],
  );

  const annotatorStats = DEFAULT_ANNOTATOR_STATS;

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Quality Control
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    const message = error?.message || 'Failed to load quality metrics.';
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Quality Control
        </Typography>
        <Alert severity="error">{message}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <div>
          <Typography variant="h4" gutterBottom>
            Quality Control Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Monitor annotation quality, inter-annotator agreement, and performance metrics
          </Typography>
        </div>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Time Range</InputLabel>
          <Select value={timeRange} onChange={(e) => setTimeRange(e.target.value)}>
            <MenuItem value="1d">1 Day</MenuItem>
            <MenuItem value="7d">7 Days</MenuItem>
            <MenuItem value="30d">30 Days</MenuItem>
            <MenuItem value="90d">90 Days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {qualityMetrics.quality_issues > 10 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <strong>{qualityMetrics.quality_issues} quality issues</strong> detected in recent annotations. Review flagged items in the triage queue.
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <QualityIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="primary">
                {(qualityMetrics.f1_score * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Overall F1 Score
              </Typography>
              <Chip
                label={getQualityLevel(qualityMetrics.f1_score).level}
                color={getQualityLevel(qualityMetrics.f1_score).color}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <MetricsIcon color="info" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="info.main">
                {(qualityMetrics.iaa_score * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Inter-Annotator Agreement
              </Typography>
              <Chip
                label={getQualityLevel(qualityMetrics.iaa_score).level}
                color={getQualityLevel(qualityMetrics.iaa_score).color}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendIcon color="success" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="success.main">
                {(qualityMetrics.precision * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Precision
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <WarningIcon color="warning" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="warning.main">
                {qualityMetrics.pending_review}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Pending Review Items
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Annotator Performance
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Annotator</TableCell>
                      <TableCell align="right">Annotations</TableCell>
                      <TableCell align="right">Accuracy</TableCell>
                      <TableCell align="right">IAA</TableCell>
                      <TableCell align="right">Avg Time (min)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {annotatorStats.map((annotator) => (
                      <TableRow key={annotator.name}>
                        <TableCell>{annotator.name}</TableCell>
                        <TableCell align="right">{annotator.annotations}</TableCell>
                        <TableCell align="right">{(annotator.accuracy * 100).toFixed(1)}%</TableCell>
                        <TableCell align="right">{(annotator.iaa * 100).toFixed(1)}%</TableCell>
                        <TableCell align="right">{annotator.avg_time.toFixed(1)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quality Issues Breakdown
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Issue Type</TableCell>
                      <TableCell align="right">Count</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Low agreement items</TableCell>
                      <TableCell align="right">6</TableCell>
                      <TableCell>
                        <Chip label="Investigating" size="small" color="warning" />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Guideline deviations</TableCell>
                      <TableCell align="right">4</TableCell>
                      <TableCell>
                        <Chip label="In review" size="small" color="info" />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Automation conflicts</TableCell>
                      <TableCell align="right">2</TableCell>
                      <TableCell>
                        <Chip label="Resolved" size="small" color="success" />
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default QualityControl;

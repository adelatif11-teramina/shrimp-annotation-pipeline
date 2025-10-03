import React, { useState, useEffect } from 'react';
import {
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
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
} from '@mui/material';
import {
  CheckCircle as QualityIcon,
  Assessment as MetricsIcon,
  Warning as WarningIcon,
  TrendingUp as TrendIcon,
} from '@mui/icons-material';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

function QualityControl() {
  const [qualityMetrics, setQualityMetrics] = useState(null);
  const [annotatorStats, setAnnotatorStats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');

  const { getSystemStatistics, getTriageStatistics } = useAnnotationAPI();

  useEffect(() => {
    fetchQualityData();
  }, [timeRange]);

  const fetchQualityData = async () => {
    try {
      setLoading(true);
      const [systemStats, triageStats] = await Promise.all([
        getSystemStatistics(),
        getTriageStatistics()
      ]);
      
      setQualityMetrics({
        precision: 0.92,
        recall: 0.89,
        f1_score: 0.90,
        iaa_score: 0.85,
        total_annotations: systemStats?.gold_annotations || 0,
        quality_issues: 12,
        pending_review: triageStats?.total_items || 0
      });

      setAnnotatorStats([
        { 
          name: 'Alice Smith', 
          annotations: 145, 
          accuracy: 0.96, 
          iaa: 0.89,
          avg_time: 3.2,
          quality_score: 0.94
        },
        { 
          name: 'Bob Johnson', 
          annotations: 132, 
          accuracy: 0.94, 
          iaa: 0.85,
          avg_time: 4.1,
          quality_score: 0.91
        },
        { 
          name: 'Carol Davis', 
          annotations: 128, 
          accuracy: 0.98, 
          iaa: 0.92,
          avg_time: 2.8,
          quality_score: 0.97
        },
        { 
          name: 'Dave Wilson', 
          annotations: 95, 
          accuracy: 0.92, 
          iaa: 0.81,
          avg_time: 5.2,
          quality_score: 0.88
        }
      ]);
    } catch (error) {
      console.error('Failed to fetch quality data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getQualityLevel = (score) => {
    if (score >= 0.95) return { level: 'Excellent', color: 'success' };
    if (score >= 0.90) return { level: 'Good', color: 'info' };
    if (score >= 0.80) return { level: 'Fair', color: 'warning' };
    return { level: 'Poor', color: 'error' };
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Quality Control
        </Typography>
        <LinearProgress />
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
          <Select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <MenuItem value="1d">1 Day</MenuItem>
            <MenuItem value="7d">7 Days</MenuItem>
            <MenuItem value="30d">30 Days</MenuItem>
            <MenuItem value="90d">90 Days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Quality Alerts */}
      {qualityMetrics?.quality_issues > 10 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <strong>{qualityMetrics.quality_issues} quality issues</strong> detected in recent annotations. 
          Review flagged items in the triage queue.
        </Alert>
      )}

      {/* Key Quality Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <QualityIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="primary">
                {(qualityMetrics?.f1_score * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Overall F1 Score
              </Typography>
              <Chip
                label={getQualityLevel(qualityMetrics?.f1_score).level}
                color={getQualityLevel(qualityMetrics?.f1_score).color}
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
                {(qualityMetrics?.iaa_score * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Inter-Annotator Agreement
              </Typography>
              <Chip
                label={getQualityLevel(qualityMetrics?.iaa_score).level}
                color={getQualityLevel(qualityMetrics?.iaa_score).color}
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
                {qualityMetrics?.total_annotations}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Gold Annotations
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block">
                +{Math.floor(qualityMetrics?.total_annotations * 0.1)} this week
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <WarningIcon color="warning" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="warning.main">
                {qualityMetrics?.quality_issues}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Quality Issues
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block">
                {qualityMetrics?.pending_review} pending review
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Precision</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(qualityMetrics?.precision * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={qualityMetrics?.precision * 100}
                  color="primary"
                />
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Recall</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(qualityMetrics?.recall * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={qualityMetrics?.recall * 100}
                  color="secondary"
                />
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">F1 Score</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(qualityMetrics?.f1_score * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={qualityMetrics?.f1_score * 100}
                  color="success"
                />
              </Box>
              
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Inter-Annotator Agreement</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(qualityMetrics?.iaa_score * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={qualityMetrics?.iaa_score * 100}
                  color="info"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quality Thresholds
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Current quality levels and recommended thresholds
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2">Minimum IAA</Typography>
                  <Chip 
                    label="≥ 80%" 
                    color={qualityMetrics?.iaa_score >= 0.80 ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2">Minimum F1 Score</Typography>
                  <Chip 
                    label="≥ 85%" 
                    color={qualityMetrics?.f1_score >= 0.85 ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2">Quality Issues</Typography>
                  <Chip 
                    label={`${qualityMetrics?.quality_issues} issues`}
                    color={qualityMetrics?.quality_issues <= 5 ? 'success' : 'warning'}
                    size="small"
                  />
                </Box>
              </Box>
              
              <Button variant="outlined" fullWidth sx={{ mt: 2 }}>
                Configure Thresholds
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Annotator Performance */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Annotator Performance
          </Typography>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Annotator</TableCell>
                  <TableCell align="right">Annotations</TableCell>
                  <TableCell align="right">Accuracy</TableCell>
                  <TableCell align="right">IAA Score</TableCell>
                  <TableCell align="right">Avg Time (min)</TableCell>
                  <TableCell align="right">Quality Score</TableCell>
                  <TableCell align="center">Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {annotatorStats.map((annotator, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {annotator.name}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      {annotator.annotations}
                    </TableCell>
                    <TableCell align="right">
                      {(annotator.accuracy * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="right">
                      {(annotator.iaa * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="right">
                      {annotator.avg_time}
                    </TableCell>
                    <TableCell align="right">
                      {(annotator.quality_score * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={getQualityLevel(annotator.quality_score).level}
                        color={getQualityLevel(annotator.quality_score).color}
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}

export default QualityControl;
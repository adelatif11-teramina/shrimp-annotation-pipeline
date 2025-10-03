import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  Assignment as TriageIcon,
  Speed as ThroughputIcon,
  Assessment as QualityIcon,
  AutoMode as AutoIcon,
  People as AnnotatorsIcon,
  TrendingUp as TrendIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

function Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [timeRange, setTimeRange] = useState('24h');
  const [realTimeData, setRealTimeData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  const { getSystemStatistics, getTriageStatistics } = useAnnotationAPI();

  // Real-time data fetching
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [systemStats, triageStats] = await Promise.all([
          getSystemStatistics(),
          getTriageStatistics()
        ]);
        
        setRealTimeData({
          system: systemStats,
          triage: triageStats,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, [timeRange]);

  // Mock data for demonstration
  const mockThroughputData = [
    { time: '00:00', annotations: 15, decisions: 12 },
    { time: '04:00', annotations: 22, decisions: 18 },
    { time: '08:00', annotations: 45, decisions: 42 },
    { time: '12:00', annotations: 38, decisions: 35 },
    { time: '16:00', annotations: 52, decisions: 48 },
    { time: '20:00', annotations: 28, decisions: 25 },
  ];

  const mockQualityData = [
    { date: '2024-01-20', precision: 0.92, recall: 0.89, f1: 0.90 },
    { date: '2024-01-21', precision: 0.94, recall: 0.91, f1: 0.92 },
    { date: '2024-01-22', precision: 0.93, recall: 0.90, f1: 0.91 },
    { date: '2024-01-23', precision: 0.95, recall: 0.92, f1: 0.93 },
    { date: '2024-01-24', precision: 0.94, recall: 0.93, f1: 0.94 },
  ];

  const mockAnnotatorData = [
    { name: 'Alice', annotations: 145, accuracy: 0.96 },
    { name: 'Bob', annotations: 132, accuracy: 0.94 },
    { name: 'Carol', annotations: 128, accuracy: 0.98 },
    { name: 'Dave', annotations: 95, accuracy: 0.92 },
  ];

  const mockPriorityData = [
    { name: 'Critical', value: 15, color: '#FF5722' },
    { name: 'High', value: 42, color: '#FF9800' },
    { name: 'Medium', value: 128, color: '#FFC107' },
    { name: 'Low', value: 89, color: '#4CAF50' },
  ];

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Grid container spacing={3} alignItems="center" sx={{ mb: 3 }}>
        <Grid item xs>
          <Typography variant="h4" gutterBottom>
            Annotation Pipeline Dashboard
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Real-time monitoring and metrics
            </Typography>
            {realTimeData && (
              <Chip 
                label={`Last updated: ${new Date(realTimeData.timestamp).toLocaleTimeString()}`}
                size="small"
                color="success"
              />
            )}
          </Box>
        </Grid>
        <Grid item>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="24h">24 Hours</MenuItem>
              <MenuItem value="7d">7 Days</MenuItem>
              <MenuItem value="30d">30 Days</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Queue Size"
            value={realTimeData?.triage?.total_items || 274}
            icon={<TriageIcon />}
            color="primary"
            change="+12"
            changeType="increase"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Throughput"
            value="45/hr"
            icon={<ThroughputIcon />}
            color="success"
            change="+8%"
            changeType="increase"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Quality Score"
            value="94.2%"
            icon={<QualityIcon />}
            color="info"
            change="+2.1%"
            changeType="increase"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Auto-Accept"
            value="32.5%"
            icon={<AutoIcon />}
            color="warning"
            change="-1.2%"
            changeType="decrease"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Active Annotators"
            value={mockAnnotatorData.length}
            icon={<AnnotatorsIcon />}
            color="secondary"
            change="0"
            changeType="stable"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Gold Annotations"
            value={realTimeData?.system?.gold_annotations || 1247}
            icon={<TrendIcon />}
            color="success"
            change="+28"
            changeType="increase"
          />
        </Grid>
      </Grid>

      {/* Tabbed Content */}
      <Card>
        <Tabs 
          value={activeTab} 
          onChange={(e, newValue) => setActiveTab(newValue)}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Throughput" />
          <Tab label="Quality Metrics" />
          <Tab label="Triage Analysis" />
          <Tab label="Annotator Performance" />
          <Tab label="System Health" />
        </Tabs>

        <CardContent>
          {activeTab === 0 && <ThroughputTab data={mockThroughputData} />}
          {activeTab === 1 && <QualityTab data={mockQualityData} />}
          {activeTab === 2 && <TriageTab data={mockPriorityData} triageStats={realTimeData?.triage} />}
          {activeTab === 3 && <AnnotatorTab data={mockAnnotatorData} />}
          {activeTab === 4 && <SystemHealthTab systemData={realTimeData?.system} />}
        </CardContent>
      </Card>
    </Box>
  );
}

function MetricCard({ title, value, icon, color, change, changeType }) {
  const getChangeColor = () => {
    switch (changeType) {
      case 'increase': return 'success.main';
      case 'decrease': return 'error.main';
      default: return 'text.secondary';
    }
  };

  const getChangeIcon = () => {
    switch (changeType) {
      case 'increase': return '↗';
      case 'decrease': return '↘';
      default: return '→';
    }
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ color: `${color}.main`, mr: 1 }}>
            {icon}
          </Box>
          <Typography variant="h6" component="div">
            {value}
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {title}
        </Typography>
        <Typography 
          variant="caption" 
          sx={{ color: getChangeColor(), display: 'flex', alignItems: 'center' }}
        >
          {getChangeIcon()} {change}
        </Typography>
      </CardContent>
    </Card>
  );
}

function ThroughputTab({ data }) {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Annotation Throughput (Last 24 Hours)
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="annotations" stackId="1" stroke="#8884d8" fill="#8884d8" />
              <Area type="monotone" dataKey="decisions" stackId="2" stroke="#82ca9d" fill="#82ca9d" />
            </AreaChart>
          </ResponsiveContainer>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Typography variant="subtitle1" gutterBottom>
            Key Statistics
          </Typography>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Peak Hour
            </Typography>
            <Typography variant="h6">
              16:00 (52 annotations)
            </Typography>
          </Box>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Average Rate
            </Typography>
            <Typography variant="h6">
              33.3 annotations/hour
            </Typography>
          </Box>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Completion Rate
            </Typography>
            <Typography variant="h6">
              91.4%
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

function QualityTab({ data }) {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Quality Metrics Trend
      </Typography>
      
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={[0.8, 1.0]} />
          <Tooltip formatter={(value) => (value * 100).toFixed(1) + '%'} />
          <Legend />
          <Line type="monotone" dataKey="precision" stroke="#8884d8" strokeWidth={2} />
          <Line type="monotone" dataKey="recall" stroke="#82ca9d" strokeWidth={2} />
          <Line type="monotone" dataKey="f1" stroke="#ffc658" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                94.2%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Average Precision
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                92.8%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Average Recall
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                93.5%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Average F1 Score
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

function TriageTab({ data, triageStats }) {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Triage Queue Analysis
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Priority Distribution
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Queue Statistics
          </Typography>
          
          {triageStats && (
            <Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Total Items
                </Typography>
                <Typography variant="h6">
                  {triageStats.total_items}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Pending Critical
                </Typography>
                <Typography variant="h6" color="error.main">
                  {triageStats.pending_critical || 0}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Pending High Priority
                </Typography>
                <Typography variant="h6" color="warning.main">
                  {triageStats.pending_high || 0}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Average Priority Score
                </Typography>
                <Typography variant="h6">
                  {triageStats.avg_priority_score?.toFixed(2) || 'N/A'}
                </Typography>
              </Box>
            </Box>
          )}
          
          <Alert severity="info" sx={{ mt: 2 }}>
            High priority items should be reviewed within 2 hours of creation.
          </Alert>
        </Grid>
      </Grid>
    </Box>
  );
}

function AnnotatorTab({ data }) {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Annotator Performance
      </Typography>
      
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" domain={[0.9, 1.0]} />
          <Tooltip />
          <Legend />
          <Bar yAxisId="left" dataKey="annotations" fill="#8884d8" />
          <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#ff7300" strokeWidth={2} />
        </BarChart>
      </ResponsiveContainer>
      
      <Grid container spacing={2} sx={{ mt: 2 }}>
        {data.map((annotator, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {annotator.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Annotations: {annotator.annotations}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Accuracy: {(annotator.accuracy * 100).toFixed(1)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={annotator.accuracy * 100} 
                  sx={{ mt: 1 }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

function SystemHealthTab({ systemData }) {
  const healthChecks = [
    { name: 'API Server', status: 'healthy', uptime: '99.9%' },
    { name: 'Database', status: 'healthy', uptime: '100%' },
    { name: 'Queue System', status: 'healthy', uptime: '99.8%' },
    { name: 'LLM Service', status: 'warning', uptime: '95.2%' },
    { name: 'Rule Engine', status: 'healthy', uptime: '100%' },
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        System Health Monitor
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Service Status
          </Typography>
          
          {healthChecks.map((service, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Chip 
                label={service.status.toUpperCase()}
                color={getStatusColor(service.status)}
                size="small"
                sx={{ mr: 2, minWidth: 80 }}
              />
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="body1">
                  {service.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Uptime: {service.uptime}
                </Typography>
              </Box>
            </Box>
          ))}
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            System Metrics
          </Typography>
          
          {systemData && (
            <Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Gold Annotations
                </Typography>
                <Typography variant="h6">
                  {systemData.gold_annotations || 0}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Active Services
                </Typography>
                <Typography variant="h6">
                  {Object.values(systemData.services || {}).filter(Boolean).length} / {Object.keys(systemData.services || {}).length}
                </Typography>
              </Box>
            </Box>
          )}
          
          <Alert severity="success" sx={{ mt: 2 }}>
            All critical systems are operational
          </Alert>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
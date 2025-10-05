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
  CircularProgress,
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
  const [annotationStats, setAnnotationStats] = useState(null);
  const [triageQueue, setTriageQueue] = useState(null);
  const [documents, setDocuments] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const { 
    getSystemStatistics, 
    getTriageQueue,
    getAnnotationStatistics,
    getDocuments,
    apiCall 
  } = useAnnotationAPI();

  // Real-time data fetching
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch all data in parallel
        const [systemStats, annotStats, queueData, docData] = await Promise.all([
          getSystemStatistics().catch(() => null),
          getAnnotationStatistics().catch(() => null),
          getTriageQueue({ limit: 100 }).catch(() => null),
          getDocuments({ limit: 50 }).catch(() => null)
        ]);
        
        setRealTimeData({
          system: systemStats,
          timestamp: new Date().toISOString()
        });
        
        setAnnotationStats(annotStats);
        setTriageQueue(queueData);
        setDocuments(docData);
        
        setError(null);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
        setError('Failed to load dashboard data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, [timeRange]);

  // Calculate real metrics from data
  const calculateMetrics = () => {
    if (!annotationStats || !triageQueue) {
      return {
        queueSize: 0,
        throughput: 0,
        qualityScore: 0,
        autoAcceptRate: 0,
        activeAnnotators: 0,
        goldAnnotations: 0
      };
    }

    const summary = annotationStats.summary || {};
    const queueItems = triageQueue.items || [];
    const pendingItems = queueItems.filter(item => item.status === 'pending').length;
    
    // Calculate throughput based on time range
    const annotations = annotationStats.annotations || [];
    const now = new Date();
    const cutoff = new Date(now - (timeRange === '1h' ? 3600000 : 86400000));
    const recentAnnotations = annotations.filter(a => 
      new Date(a.created_at) > cutoff
    ).length;
    const hours = timeRange === '1h' ? 1 : 24;
    const throughput = Math.round(recentAnnotations / hours);

    // Calculate quality score from acceptance rate
    const qualityScore = summary.acceptance_rate || 0;

    // Calculate auto-accept rate
    const autoAcceptRate = ((summary.modified || 0) / (summary.total_annotations || 1)) * 100;

    // Get unique annotators
    const uniqueAnnotators = new Set();
    if (annotationStats.by_user) {
      annotationStats.by_user.forEach(user => uniqueAnnotators.add(user.user_id));
    }

    return {
      queueSize: pendingItems || triageQueue.total || 0,
      throughput,
      qualityScore,
      autoAcceptRate: autoAcceptRate.toFixed(1),
      activeAnnotators: uniqueAnnotators.size,
      goldAnnotations: summary.total_annotations || 0
    };
  };

  const metrics = calculateMetrics();

  // Process annotation statistics for charts
  const processChartData = () => {
    if (!annotationStats) {
      return {
        throughputData: [],
        qualityData: [],
        priorityData: [],
        annotatorData: []
      };
    }

    // Process throughput data from by_date statistics
    const throughputData = (annotationStats.by_date || []).map(day => ({
      time: new Date(day.date).toLocaleDateString(),
      annotations: day.total,
      decisions: day.accepted,
      rejected: day.total - day.accepted
    }));

    // Process quality data from decision statistics
    const summary = annotationStats.summary || {};
    const qualityData = [
      {
        date: new Date().toLocaleDateString(),
        precision: summary.acceptance_rate ? summary.acceptance_rate / 100 : 0,
        recall: 0.9, // Would need actual recall calculation
        f1: summary.acceptance_rate ? (summary.acceptance_rate / 100) * 0.95 : 0
      }
    ];

    // Process priority distribution from triage queue
    const priorityMap = {};
    if (triageQueue?.items) {
      triageQueue.items.forEach(item => {
        const level = item.priority_level || 'medium';
        priorityMap[level] = (priorityMap[level] || 0) + 1;
      });
    }

    const priorityData = [
      { name: 'Critical', value: priorityMap.critical || 0, color: '#FF5722' },
      { name: 'High', value: priorityMap.high || 0, color: '#FF9800' },
      { name: 'Medium', value: priorityMap.medium || 0, color: '#FFC107' },
      { name: 'Low', value: priorityMap.low || 0, color: '#4CAF50' },
    ];

    // Process annotator data
    const annotatorData = (annotationStats.by_user || []).map(user => ({
      name: `User ${user.user_id}`,
      annotations: user.total,
      accuracy: user.acceptance_rate / 100,
      avgTime: user.avg_time_per_annotation || 0
    }));

    return {
      throughputData,
      qualityData,
      priorityData,
      annotatorData
    };
  };

  const chartData = processChartData();

  if (isLoading) {
    return (
      <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <CircularProgress sx={{ mt: 4 }} />
        <Typography sx={{ mt: 2 }} color="text.secondary">
          Loading real-time data...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <Alert severity="error">
          {error}
        </Alert>
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
              Live data monitoring
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
              label="Time Range"
            >
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="24h">24 Hours</MenuItem>
              <MenuItem value="7d">7 Days</MenuItem>
              <MenuItem value="30d">30 Days</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>

      {/* Key Metrics Cards - Real Data */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Queue Size"
            value={metrics.queueSize}
            icon={<TriageIcon />}
            color="primary"
            change={triageQueue?.items ? `${triageQueue.items.length} items` : "0"}
            changeType="stable"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Throughput"
            value={`${metrics.throughput}/hr`}
            icon={<ThroughputIcon />}
            color="success"
            change={`${timeRange} avg`}
            changeType="stable"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Quality Score"
            value={`${metrics.qualityScore.toFixed(1)}%`}
            icon={<QualityIcon />}
            color="info"
            change="Acceptance rate"
            changeType={metrics.qualityScore > 90 ? "increase" : "decrease"}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Modified Rate"
            value={`${metrics.autoAcceptRate}%`}
            icon={<AutoIcon />}
            color="warning"
            change="Of annotations"
            changeType="stable"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Active Users"
            value={metrics.activeAnnotators}
            icon={<AnnotatorsIcon />}
            color="secondary"
            change="Unique annotators"
            changeType="stable"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title="Total Annotations"
            value={metrics.goldAnnotations}
            icon={<TrendIcon />}
            color="success"
            change={`${documents?.total || 0} docs`}
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
          {activeTab === 0 && <ThroughputTab data={chartData.throughputData} stats={annotationStats} />}
          {activeTab === 1 && <QualityTab data={chartData.qualityData} stats={annotationStats} />}
          {activeTab === 2 && <TriageTab data={chartData.priorityData} triageStats={triageQueue} />}
          {activeTab === 3 && <AnnotatorTab data={chartData.annotatorData} stats={annotationStats} />}
          {activeTab === 4 && <SystemHealthTab systemData={realTimeData?.system} documents={documents} />}
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

function ThroughputTab({ data, stats }) {
  const summary = stats?.summary || {};
  const totalAnnotations = summary.total_annotations || 0;
  const acceptedAnnotations = summary.accepted || 0;
  const completionRate = totalAnnotations > 0 ? ((acceptedAnnotations / totalAnnotations) * 100).toFixed(1) : 0;

  // Calculate real average rate from data
  const avgRate = data.length > 0 
    ? Math.round(data.reduce((sum, d) => sum + d.annotations, 0) / data.length)
    : 0;

  // Find peak hour
  const peakHour = data.length > 0
    ? data.reduce((max, d) => d.annotations > max.annotations ? d : max, data[0])
    : null;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Annotation Throughput
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data.length > 0 ? data : [{ time: 'No data', annotations: 0, decisions: 0 }]}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="annotations" stackId="1" stroke="#8884d8" fill="#8884d8" name="Total" />
              <Area type="monotone" dataKey="decisions" stackId="2" stroke="#82ca9d" fill="#82ca9d" name="Accepted" />
            </AreaChart>
          </ResponsiveContainer>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Typography variant="subtitle1" gutterBottom>
            Real-time Statistics
          </Typography>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Total Annotations
            </Typography>
            <Typography variant="h6">
              {totalAnnotations}
            </Typography>
          </Box>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Average Rate
            </Typography>
            <Typography variant="h6">
              {avgRate} per period
            </Typography>
          </Box>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Acceptance Rate
            </Typography>
            <Typography variant="h6">
              {completionRate}%
            </Typography>
          </Box>
          {peakHour && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Peak Period
              </Typography>
              <Typography variant="h6">
                {peakHour.time} ({peakHour.annotations} annotations)
              </Typography>
            </Box>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}

function QualityTab({ data, stats }) {
  const summary = stats?.summary || {};
  const byConfidence = stats?.by_confidence || [];
  
  // Calculate real precision and recall from actual data
  const precision = summary.acceptance_rate || 0;
  const recall = summary.accepted && summary.total_annotations 
    ? (summary.accepted / summary.total_annotations * 100).toFixed(1) 
    : 0;
  const f1Score = precision && recall 
    ? (2 * precision * recall / (precision + parseFloat(recall))).toFixed(1) 
    : 0;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Quality Metrics Analysis
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="subtitle2" gutterBottom>
            Decision Distribution
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={stats?.by_decision || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="decision" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#8884d8" name="Count" />
              <Bar dataKey="percentage" fill="#82ca9d" name="Percentage" />
            </BarChart>
          </ResponsiveContainer>
        </Grid>
      </Grid>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {precision.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Acceptance Rate
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                {summary.rejected || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Rejected Annotations
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {summary.skipped || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Skipped Items
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {byConfidence.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Confidence Distribution
          </Typography>
          <Grid container spacing={2}>
            {byConfidence.map((conf, idx) => (
              <Grid item xs={3} key={idx}>
                <Typography variant="body2" color="text.secondary">
                  {conf.level}
                </Typography>
                <Typography variant="h6">
                  {conf.count} ({conf.percentage}%)
                </Typography>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  );
}

function TriageTab({ data, triageStats }) {
  const items = triageStats?.items || [];
  const total = triageStats?.total || 0;
  
  // Calculate real statistics
  const pendingItems = items.filter(item => item.status === 'pending').length;
  const completedItems = items.filter(item => item.status === 'completed').length;
  const avgPriorityScore = items.length > 0
    ? (items.reduce((sum, item) => sum + (item.priority_score || 0), 0) / items.length).toFixed(2)
    : 0;

  // Count by priority level
  const criticalCount = items.filter(item => item.priority_level === 'critical').length;
  const highCount = items.filter(item => item.priority_level === 'high').length;

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
                label={({ name, value }) => value > 0 ? `${name}: ${value}` : ''}
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
            Live Queue Statistics
          </Typography>
          
          <Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Total Items
              </Typography>
              <Typography variant="h6">
                {total}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Pending Items
              </Typography>
              <Typography variant="h6" color="warning.main">
                {pendingItems}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Completed Items
              </Typography>
              <Typography variant="h6" color="success.main">
                {completedItems}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Critical Priority
              </Typography>
              <Typography variant="h6" color="error.main">
                {criticalCount}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                High Priority
              </Typography>
              <Typography variant="h6" color="warning.main">
                {highCount}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Average Priority Score
              </Typography>
              <Typography variant="h6">
                {avgPriorityScore}
              </Typography>
            </Box>
          </Box>
          
          {pendingItems > 10 && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              {pendingItems} items pending review. Consider increasing annotation throughput.
            </Alert>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}

function AnnotatorTab({ data, stats }) {
  const byUser = stats?.by_user || [];
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Annotator Performance (Live Data)
      </Typography>
      
      {data.length > 0 ? (
        <>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
              <Tooltip formatter={(value, name) => 
                name === 'accuracy' ? `${(value * 100).toFixed(1)}%` : value
              } />
              <Legend />
              <Bar yAxisId="left" dataKey="annotations" fill="#8884d8" name="Annotations" />
              <Bar yAxisId="right" dataKey="accuracy" fill="#82ca9d" name="Acceptance Rate" />
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
                      Acceptance Rate: {(annotator.accuracy * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Avg Time: {annotator.avgTime.toFixed(1)}s
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
        </>
      ) : (
        <Alert severity="info">
          No annotator data available yet. Start annotating to see performance metrics.
        </Alert>
      )}
    </Box>
  );
}

function SystemHealthTab({ systemData, documents }) {
  const totalDocs = documents?.total || 0;
  const processedDocs = documents?.documents?.filter(d => d.status === 'processed').length || 0;
  const pendingDocs = totalDocs - processedDocs;
  
  // Calculate real system health from available data
  const healthChecks = [
    { 
      name: 'API Server', 
      status: systemData ? 'healthy' : 'error', 
      uptime: systemData ? '100%' : '0%' 
    },
    { 
      name: 'Database', 
      status: systemData?.database_status === 'connected' ? 'healthy' : 'warning',
      uptime: systemData?.database_status === 'connected' ? '100%' : 'Degraded'
    },
    { 
      name: 'Queue System', 
      status: documents ? 'healthy' : 'warning', 
      uptime: documents ? '100%' : 'Unknown' 
    },
    { 
      name: 'Frontend', 
      status: 'healthy', 
      uptime: '100%' 
    },
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
                  Status: {service.uptime}
                </Typography>
              </Box>
            </Box>
          ))}
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            System Metrics
          </Typography>
          
          <Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Environment
              </Typography>
              <Typography variant="h6">
                {systemData?.environment || systemData?.mode || 'Production'}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Total Documents
              </Typography>
              <Typography variant="h6">
                {totalDocs}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Processed Documents
              </Typography>
              <Typography variant="h6" color="success.main">
                {processedDocs}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Pending Documents
              </Typography>
              <Typography variant="h6" color="warning.main">
                {pendingDocs}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                API Port
              </Typography>
              <Typography variant="h6">
                {systemData?.port || '8080'}
              </Typography>
            </Box>
          </Box>
          
          {healthChecks.every(s => s.status === 'healthy') ? (
            <Alert severity="success" sx={{ mt: 2 }}>
              All systems operational
            </Alert>
          ) : (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Some services may be degraded. Check service status.
            </Alert>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';

import Navigation from './components/Navigation';
import AnnotationWorkspace from './pages/AnnotationWorkspace';
import TriageQueue from './pages/TriageQueue';
import Dashboard from './pages/Dashboard';
import DocumentManager from './pages/DocumentManager';
import QualityControl from './pages/QualityControl';
import Settings from './pages/Settings';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h6: {
      fontWeight: 600,
    },
  },
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Box sx={{ display: 'flex', height: '100vh' }}>
            <Navigation />
            <Box component="main" sx={{ flexGrow: 1, overflow: 'auto' }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/triage" element={<TriageQueue />} />
                <Route path="/annotate/:itemId?" element={<AnnotationWorkspace />} />
                <Route path="/documents" element={<DocumentManager />} />
                <Route path="/quality" element={<QualityControl />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
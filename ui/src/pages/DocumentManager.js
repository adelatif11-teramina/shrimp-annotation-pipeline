import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
} from '@mui/material';
import {
  Upload as UploadIcon,
  Visibility as ViewIcon,
  GetApp as ExportIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

function DocumentManager() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploadDialog, setUploadDialog] = useState(false);
  const [newDocument, setNewDocument] = useState({
    title: '',
    text: '',
    source: 'manual'
  });
  const [selectedFile, setSelectedFile] = useState(null);

  const { getDocuments, ingestDocument } = useAnnotationAPI();

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await getDocuments();
      // Extract documents array from API response
      setDocuments(response?.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      setDocuments([]); // Ensure documents is always an array on error
    } finally {
      setLoading(false);
    }
  };

  const extractTextFromPDF = async (file) => {
    try {
      // Simple PDF text extraction using browser APIs
      const arrayBuffer = await file.arrayBuffer();
      const text = await extractPDFText(arrayBuffer);
      return text;
    } catch (error) {
      console.error('PDF extraction failed:', error);
      return null;
    }
  };

  const extractPDFText = async (arrayBuffer) => {
    // Basic PDF text extraction - this is a simplified approach
    // For production, you'd want to use a proper PDF.js implementation
    const decoder = new TextDecoder('utf-8');
    const text = decoder.decode(arrayBuffer);
    
    // Extract readable text between 'stream' and 'endstream' markers
    const textContent = [];
    const streamRegex = /stream\s*(.*?)\s*endstream/gs;
    let match;
    
    while ((match = streamRegex.exec(text)) !== null) {
      const streamContent = match[1];
      // Remove PDF formatting and extract readable text
      const readableText = streamContent
        .replace(/[^\x20-\x7E\n\r]/g, ' ') // Keep only printable ASCII
        .replace(/\s+/g, ' ') // Normalize whitespace
        .trim();
      
      if (readableText.length > 10) { // Only include substantial content
        textContent.push(readableText);
      }
    }
    
    return textContent.join('\n\n').trim();
  };

  const handleFileSelect = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Set title to filename if not already set
      if (!newDocument.title) {
        setNewDocument({...newDocument, title: file.name.replace(/\.[^/.]+$/, "")});
      }
      
      // Handle different file types
      if (file.type === 'text/plain') {
        const reader = new FileReader();
        reader.onload = (e) => {
          setNewDocument({...newDocument, text: e.target.result});
        };
        reader.readAsText(file);
      } else if (file.type === 'application/pdf') {
        // Attempt PDF text extraction
        const extractedText = await extractTextFromPDF(file);
        if (extractedText) {
          setNewDocument({...newDocument, text: extractedText});
        } else {
          // If extraction fails, show a helpful message
          setNewDocument({
            ...newDocument, 
            text: `// PDF uploaded: ${file.name}\n// Please manually copy and paste the text content from your PDF here,\n// or use a PDF-to-text converter and upload as a .txt file.`
          });
        }
      }
    }
  };

  const handleUploadDocument = async () => {
    try {
      let documentText = newDocument.text;
      
      // If uploading a file and no text is set, read the file
      if (selectedFile && !documentText) {
        if (selectedFile.type === 'text/plain') {
          const reader = new FileReader();
          reader.onload = async (e) => {
            const docData = {
              doc_id: `doc_${Date.now()}`,
              title: newDocument.title,
              text: e.target.result,
              source: newDocument.source,
              metadata: { 
                original_filename: selectedFile.name,
                file_type: selectedFile.type 
              }
            };
            await ingestDocument(docData);
            resetForm();
            fetchDocuments();
          };
          reader.readAsText(selectedFile);
          return;
        } else {
          alert('Currently only text files are supported. For PDFs, please extract the text first.');
          return;
        }
      }

      const docData = {
        doc_id: `doc_${Date.now()}`,
        title: newDocument.title,
        text: documentText,
        source: newDocument.source,
        metadata: selectedFile ? { 
          original_filename: selectedFile.name,
          file_type: selectedFile.type 
        } : {}
      };

      await ingestDocument(docData);
      resetForm();
      fetchDocuments();
    } catch (error) {
      console.error('Failed to upload document:', error);
    }
  };

  const resetForm = () => {
    setUploadDialog(false);
    setNewDocument({ title: '', text: '', source: 'manual' });
    setSelectedFile(null);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'processed': return 'success';
      case 'processing': return 'warning';
      case 'pending': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Document Manager
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
            Document Manager
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage documents for annotation and training data generation
          </Typography>
        </div>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setUploadDialog(true)}
        >
          Add Document
        </Button>
      </Box>

      {/* Document Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {documents.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Documents
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                {documents.filter(doc => doc.status === 'processed').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Processed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {documents.filter(doc => doc.status === 'processing').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Processing
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="text.secondary">
                {documents.filter(doc => doc.status === 'pending').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Pending
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Documents Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Document ID</TableCell>
              <TableCell>Title</TableCell>
              <TableCell>Source</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Sentences</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {documents.map((doc, index) => (
              <TableRow key={doc.id || index}>
                <TableCell>
                  <Typography variant="body2" fontFamily="monospace">
                    {doc.doc_id || `doc_${index}`}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" fontWeight="medium">
                    {doc.title || 'Untitled Document'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={doc.source || 'unknown'}
                    size="small"
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={doc.status || 'pending'}
                    color={getStatusColor(doc.status)}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {doc.created_at 
                      ? new Date(doc.created_at).toLocaleDateString()
                      : new Date().toLocaleDateString()
                    }
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {doc.sentence_count || '0'} sentences
                  </Typography>
                </TableCell>
                <TableCell>
                  <IconButton color="primary" size="small">
                    <ViewIcon />
                  </IconButton>
                  <IconButton color="default" size="small">
                    <ExportIcon />
                  </IconButton>
                  <IconButton color="error" size="small">
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {documents.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h6" color="text.secondary">
            No documents found
          </Typography>
          <Typography variant="body2" color="text.secondary" mt={1}>
            Upload your first document to get started with annotation.
          </Typography>
        </Box>
      )}

      {/* Upload Dialog */}
      <Dialog open={uploadDialog} onClose={() => setUploadDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Add New Document</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Document Title"
                value={newDocument.title}
                onChange={(e) => setNewDocument({...newDocument, title: e.target.value})}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Source"
                value={newDocument.source}
                onChange={(e) => setNewDocument({...newDocument, source: e.target.value})}
                SelectProps={{ native: true }}
              >
                <option value="manual">Manual Entry</option>
                <option value="pdf">PDF Upload</option>
                <option value="paper">Research Paper</option>
                <option value="report">Technical Report</option>
                <option value="hatchery_log">Hatchery Log</option>
              </TextField>
            </Grid>
            <Grid item xs={12}>
              {newDocument.source === 'pdf' ? (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Upload File
                  </Typography>
                  <input
                    type="file"
                    accept=".txt,.pdf"
                    onChange={handleFileSelect}
                    style={{ marginBottom: '16px' }}
                  />
                  {selectedFile && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Selected: {selectedFile.name}
                    </Typography>
                  )}
                  <TextField
                    fullWidth
                    multiline
                    rows={6}
                    label="Document Text (auto-filled from file or paste manually)"
                    value={newDocument.text}
                    onChange={(e) => setNewDocument({...newDocument, text: e.target.value})}
                    placeholder="Text will appear here when you select a file, or you can paste text manually..."
                  />
                </Box>
              ) : (
                <TextField
                  fullWidth
                  multiline
                  rows={8}
                  label="Document Text"
                  value={newDocument.text}
                  onChange={(e) => setNewDocument({...newDocument, text: e.target.value})}
                  placeholder="Paste or type your document text here..."
                />
              )}
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={resetForm}>Cancel</Button>
          <Button 
            onClick={handleUploadDocument}
            variant="contained"
            disabled={!newDocument.title || (!newDocument.text && !selectedFile)}
          >
            Upload Document
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default DocumentManager;
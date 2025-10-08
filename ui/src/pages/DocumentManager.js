import React, { useMemo, useState } from 'react';
import {
  Alert,
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
  Snackbar,
  Tooltip,
} from '@mui/material';
import {
  Upload as UploadIcon,
  Visibility as ViewIcon,
  GetApp as ExportIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';

import { useAnnotationAPI } from '../hooks/useAnnotationAPI';

const readFileAsText = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(event.target?.result || '');
    reader.onerror = (event) => reject(event?.target?.error || new Error('Failed to read file'));
    reader.readAsText(file);
  });

const extractPDFText = async (arrayBuffer) => {
  const decoder = new TextDecoder('utf-8');
  const text = decoder.decode(arrayBuffer);

  const textContent = [];
  const streamRegex = /stream\s*(.*?)\s*endstream/gs;
  let match;

  while ((match = streamRegex.exec(text)) !== null) {
    const streamContent = match[1];
    const readableText = streamContent
      .replace(/[^\x20-\x7E\n\r]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (readableText.length > 10) {
      textContent.push(readableText);
    }
  }

  return textContent.join('\n\n').trim();
};

function DocumentManager() {
  const [uploadDialog, setUploadDialog] = useState(false);
  const [newDocument, setNewDocument] = useState({ title: '', text: '', source: 'manual' });
  const [selectedFile, setSelectedFile] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const queryClient = useQueryClient();
  const { getDocuments, ingestDocument } = useAnnotationAPI();

  const documentsQuery = useQuery(['documents', { limit: 100 }], () => getDocuments({ limit: 100 }), {
    keepPreviousData: true,
  });

  const ingestMutation = useMutation(ingestDocument, {
    onSuccess: () => {
      queryClient.invalidateQueries(['documents']);
      handleSnackbar('Document ingested successfully.', 'success');
      resetForm();
    },
    onError: (error) => {
      const message = error?.message || 'Failed to upload document.';
      handleSnackbar(message, 'error');
    },
  });

  const documents = useMemo(() => documentsQuery.data?.documents || [], [documentsQuery.data]);

  const handleSnackbar = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const closeSnackbar = (_, reason) => {
    if (reason === 'clickaway') return;
    setSnackbar((prev) => ({ ...prev, open: false }));
  };

  const handleFileSelect = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setSelectedFile(file);

    if (!newDocument.title) {
      setNewDocument((prev) => ({ ...prev, title: file.name.replace(/\.[^/.]+$/, '') }));
    }

    if (file.type === 'text/plain') {
      const text = await readFileAsText(file);
      setNewDocument((prev) => ({ ...prev, text }));
      return;
    }

    if (file.type === 'application/pdf') {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const text = await extractPDFText(arrayBuffer);
        if (text) {
          setNewDocument((prev) => ({ ...prev, text }));
        } else {
          setNewDocument((prev) => ({
            ...prev,
            text: `// PDF uploaded: ${file.name}\n// Please paste the extracted text here or upload a .txt file instead.`,
          }));
          handleSnackbar('PDF text extraction is limited. Please verify the imported content.', 'warning');
        }
      } catch (error) {
        console.error('PDF extraction failed:', error);
        handleSnackbar('Unable to extract text from PDF. Please upload a .txt file.', 'warning');
      }
      return;
    }

    handleSnackbar('Only plain text or PDF files are supported.', 'warning');
  };

  const handleUploadDocument = async () => {
    try {
      let documentText = newDocument.text;

      if (selectedFile && !documentText) {
        if (selectedFile.type === 'text/plain') {
          documentText = await readFileAsText(selectedFile);
        } else {
          handleSnackbar('Please extract text from the selected file before uploading.', 'warning');
          return;
        }
      }

      if (!newDocument.title || !documentText) {
        handleSnackbar('Title and document text are required.', 'warning');
        return;
      }

      const docData = {
        doc_id: `doc_${Date.now()}`,
        title: newDocument.title,
        text: documentText,
        source: newDocument.source,
        metadata: selectedFile
          ? {
              original_filename: selectedFile.name,
              file_type: selectedFile.type,
            }
          : {},
      };

      await ingestMutation.mutateAsync(docData);
    } catch (error) {
      console.error('Failed to upload document:', error);
      handleSnackbar('Failed to upload document.', 'error');
    }
  };

  const resetForm = () => {
    setUploadDialog(false);
    setNewDocument({ title: '', text: '', source: 'manual' });
    setSelectedFile(null);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'processed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'pending':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const isLoading = documentsQuery.isLoading;
  const isRefetching = documentsQuery.isFetching;

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <div>
          <Typography variant="h4" gutterBottom>
            Document Manager
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage source documents available to the annotation pipeline.
          </Typography>
        </div>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setUploadDialog(true)}
        >
          Upload Document
        </Button>
      </Box>

      {isLoading ? (
        <Box sx={{ py: 5, textAlign: 'center' }}>
          <LinearProgress sx={{ maxWidth: 320, mx: 'auto', mb: 2 }} />
          <Typography color="text.secondary">Loading documents…</Typography>
        </Box>
      ) : (
        <>
          {isRefetching && <LinearProgress sx={{ mb: 2 }} />}

          <Card>
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={6} md={3}>
                  <Typography variant="h5">{documents.length}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total documents
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="h5">
                    {documents.filter((doc) => doc.status === 'processed').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processed
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="h5">
                    {documents.filter((doc) => doc.status === 'pending').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Pending
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="h5">
                    {documents.filter((doc) => doc.status === 'error').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    With errors
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          <TableContainer component={Paper} sx={{ mt: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Title</TableCell>
                  <TableCell>Source</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {documents.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center">
                      <Typography color="text.secondary">
                        No documents available yet. Upload one to get started.
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  documents.map((doc) => (
                    <TableRow key={doc.doc_id || doc.id} hover>
                      <TableCell>
                        <Typography variant="body1" fontWeight={500}>
                          {doc.title || 'Untitled'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {doc.doc_id || doc.id}
                        </Typography>
                      </TableCell>
                      <TableCell>{doc.source || 'unknown'}</TableCell>
                      <TableCell>
                        <Chip
                          label={doc.status || 'pending'}
                          size="small"
                          color={getStatusColor(doc.status)}
                          variant={doc.status === 'processed' ? 'filled' : 'outlined'}
                        />
                      </TableCell>
                      <TableCell>
                        {doc.created_at
                          ? new Date(doc.created_at).toLocaleString()
                          : '—'}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="Preview document">
                          <span>
                            <IconButton disabled={!doc.text}>
                              <ViewIcon />
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Tooltip title="Export document">
                          <IconButton>
                            <ExportIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete document">
                          <IconButton color="error">
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}

      <Dialog open={uploadDialog} onClose={resetForm} fullWidth maxWidth="md">
        <DialogTitle>Upload Document</DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Title"
                value={newDocument.title}
                onChange={(e) => setNewDocument((prev) => ({ ...prev, title: e.target.value }))}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Source"
                value={newDocument.source}
                onChange={(e) => setNewDocument((prev) => ({ ...prev, source: e.target.value }))}
                sx={{ mb: 2 }}
              />
              <Button
                component="label"
                variant="outlined"
                startIcon={<UploadIcon />}
              >
                Select File
                <input type="file" hidden onChange={handleFileSelect} />
              </Button>
              {selectedFile && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Selected: {selectedFile.name}
                </Typography>
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                multiline
                minRows={10}
                label="Document Text"
                value={newDocument.text}
                onChange={(e) => setNewDocument((prev) => ({ ...prev, text: e.target.value }))}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={resetForm}>Cancel</Button>
          <Button
            variant="contained"
            startIcon={<UploadIcon />}
            onClick={handleUploadDocument}
            disabled={ingestMutation.isLoading}
          >
            {ingestMutation.isLoading ? 'Uploading…' : 'Upload'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={closeSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={closeSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default DocumentManager;

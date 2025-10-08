import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Chip,
  Box,
} from '@mui/material';

function AnnotationGuidelinesDialog({ open, onClose, entityTypes }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Annotation Guidelines</DialogTitle>
      <DialogContent>
        <Typography variant="body1" paragraph>
          Quick reference for shrimp aquaculture annotation:
        </Typography>

        <Typography variant="h6" gutterBottom>
          Entity Types
        </Typography>
        {Object.entries(entityTypes).map(([type, config]) => (
          <Box key={type} sx={{ mb: 1 }}>
            <Chip
              label={type}
              sx={{ backgroundColor: config.color, color: 'white', mr: 1 }}
              size="small"
            />
            <Typography variant="body2" component="span">
              {config.description}
            </Typography>
          </Box>
        ))}

        <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
          Key Guidelines
        </Typography>
        <Typography variant="body2" component="div">
          <ul>
            <li>Select text precisely - include only the entity span</li>
            <li>Use canonical forms when possible (e.g., "Penaeus vannamei" not "white shrimp")</li>
            <li>Only annotate explicit relationships, not implied ones</li>
            <li>Mark uncertain annotations with low confidence</li>
            <li>Add notes for ambiguous cases</li>
          </ul>
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default AnnotationGuidelinesDialog;

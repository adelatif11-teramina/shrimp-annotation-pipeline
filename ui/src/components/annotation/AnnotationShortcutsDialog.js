import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Grid,
} from '@mui/material';

function AnnotationShortcutsDialog({ open, onClose }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Keyboard Shortcuts</DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Actions
            </Typography>
            <Box sx={{ '& > div': { mb: 1 } }}>
              <ShortcutRow combo="Ctrl+Enter" label="Accept" />
              <ShortcutRow combo="Ctrl+R" label="Reject" />
              <ShortcutRow combo="Ctrl+M" label="Modify" />
              <ShortcutRow combo="Ctrl+Shift+S" label="Skip" />
              <ShortcutRow combo="Ctrl+N" label="Next Item" />
              <ShortcutRow combo="Ctrl+S" label="Save Progress" />
              <ShortcutRow combo="Esc" label="Close Dialog" />
            </Box>
          </Grid>

          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Modes & Types
            </Typography>
            <Box sx={{ '& > div': { mb: 1 } }}>
              <ShortcutRow combo="E" label="Entity Mode" />
              <ShortcutRow combo="R" label="Relation Mode" />
              <ShortcutRow combo="T" label="Topic Mode" />
            </Box>

            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Entity Types
            </Typography>
            <Box sx={{ '& > div': { mb: 1 } }}>
              <ShortcutRow combo="1" label="SPECIES" />
              <ShortcutRow combo="2" label="PATHOGEN" />
              <ShortcutRow combo="3" label="DISEASE" />
              <ShortcutRow combo="4" label="CLINICAL_SYMPTOM" />
              <ShortcutRow combo="5" label="PHENOTYPIC_TRAIT" />
              <ShortcutRow combo="6" label="GENE" />
              <ShortcutRow combo="7" label="CHEMICAL_COMPOUND" />
              <ShortcutRow combo="8" label="TREATMENT" />
              <ShortcutRow combo="9" label="LIFE_STAGE" />
              <ShortcutRow combo="0" label="MEASUREMENT" />
            </Box>

            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Help
            </Typography>
            <Box sx={{ '& > div': { mb: 1 } }}>
              <ShortcutRow combo="? or H" label="Show this help" />
              <ShortcutRow combo="F1" label="Show guidelines" />
            </Box>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

function ShortcutRow({ combo, label }) {
  return (
    <Box display="flex" justifyContent="space-between">
      <Typography variant="body2">
        <strong>{combo}</strong>
      </Typography>
      <Typography variant="body2">{label}</Typography>
    </Box>
  );
}

export default AnnotationShortcutsDialog;

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Chip,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Grid,
  Button,
} from '@mui/material';
import AcceptIcon from '@mui/icons-material/Check';
import EditIcon from '@mui/icons-material/Edit';
import RejectIcon from '@mui/icons-material/Close';
import SkipIcon from '@mui/icons-material/SkipNext';

function AnnotationSidebar({
  currentItem,
  confidence,
  onConfidenceChange,
  notes,
  onNotesChange,
  isSubmitting,
  onAccept,
  onModify,
  onReject,
  onSkip,
  triplets,
}) {
  const candidateEntities = currentItem?.candidate_data?.entities || [];
  const ruleEntities = currentItem?.rule_results?.entities || [];
  const candidateTriplets = triplets || currentItem?.candidate_data?.triplets || [];

  return (
    <Grid container direction="column" spacing={2}>
      <Grid item>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              AI Suggestions
            </Typography>
            {candidateEntities.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Entities
                </Typography>
                {candidateEntities.map((entity, index) => (
                  <Chip
                    key={index}
                    label={`${entity.text} (${entity.label})`}
                    size="small"
                    sx={{ m: 0.5 }}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            )}

            {ruleEntities.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Rule-based
                </Typography>
                {ruleEntities.map((entity, index) => (
                  <Chip
                    key={index}
                    label={`${entity.text} (${entity.label})`}
                    size="small"
                    sx={{ m: 0.5 }}
                    color="secondary"
                    variant="outlined"
                  />
                ))}
              </Box>
            )}

            {candidateEntities.length === 0 && ruleEntities.length === 0 && (
              <Typography variant="body2" color="text.secondary">
                No suggestions available for this item.
              </Typography>
            )}

            {candidateTriplets.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Triplet Candidates
                </Typography>
                {candidateTriplets.slice(0, 4).map((triplet) => (
                  <Chip
                    key={triplet.triplet_id}
                    label={`${triplet.head?.text || '…'} — ${triplet.relation || '…'} → ${triplet.tail?.text || '…'}`}
                    size="small"
                    sx={{ m: 0.5 }}
                    color={triplet.reviewer_action === 'approve' ? 'success' : 'default'}
                    variant="outlined"
                  />
                ))}
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Annotation
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Confidence</InputLabel>
              <Select value={confidence} onChange={(event) => onConfidenceChange(event.target.value)}>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Notes"
              multiline
              rows={3}
              fullWidth
              value={notes}
              onChange={(event) => onNotesChange(event.target.value)}
              placeholder="Add notes about this annotation..."
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Actions
            </Typography>

            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant="contained"
                  color="success"
                  startIcon={<AcceptIcon />}
                  onClick={onAccept}
                  disabled={isSubmitting}
                >
                  Accept
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant="contained"
                  color="warning"
                  startIcon={<EditIcon />}
                  onClick={onModify}
                  disabled={isSubmitting}
                >
                  Modify
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant="outlined"
                  color="error"
                  startIcon={<RejectIcon />}
                  onClick={onReject}
                  disabled={isSubmitting}
                >
                  Reject
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<SkipIcon />}
                  onClick={onSkip}
                  disabled={isSubmitting}
                >
                  Skip
                </Button>
              </Grid>
            </Grid>

            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Keyboard shortcuts:
              </Typography>
              <Typography variant="caption" display="block">
                Ctrl+Enter: Accept | Ctrl+R: Reject
              </Typography>
              <Typography variant="caption" display="block">
                Ctrl+S: Save | Ctrl+N: Next
              </Typography>
              <Typography variant="caption" display="block">
                1-0: Entity types | ?: Help
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

export default AnnotationSidebar;

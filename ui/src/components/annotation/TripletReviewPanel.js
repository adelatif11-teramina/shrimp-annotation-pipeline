import React from 'react';
import PropTypes from 'prop-types';
import {
  Card,
  CardContent,
  Typography,
  Stack,
  Grid,
  Chip,
  ButtonGroup,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import EditIcon from '@mui/icons-material/Edit';
import SkipNextIcon from '@mui/icons-material/SkipNext';

const actionButtons = [
  { value: 'approve', label: 'Approve', icon: <CheckIcon fontSize="small" />, color: 'success' },
  { value: 'reject', label: 'Reject', icon: <CloseIcon fontSize="small" />, color: 'error' },
  { value: 'revise', label: 'Needs Edit', icon: <EditIcon fontSize="small" />, color: 'warning' },
  { value: 'skip', label: 'Skip', icon: <SkipNextIcon fontSize="small" />, color: 'primary' },
];

function TripletReviewPanel({ triplets, relationTypes, onUpdate }) {
  if (!triplets || triplets.length === 0) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            No triplet candidates available for this sentence.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const relationOptions = relationTypes.map((relation) => relation.toUpperCase());

  return (
    <Stack spacing={2} sx={{ overflowY: 'auto', pr: 1 }}>
      {triplets.map((triplet) => {
        const currentAction = triplet.reviewer_action || 'pending';
        const auditStatus = triplet.audit?.status || 'n/a';
        const ruleChipColor = triplet.rule_support ? 'success' : 'default';
        const head = triplet.edited?.head || triplet.head || {};
        const tail = triplet.edited?.tail || triplet.tail || {};
        const relationValue = (triplet.edited?.relation || triplet.relation || '').toUpperCase();
        const evidence = triplet.edited?.evidence ?? triplet.evidence ?? '';

        return (
          <Card key={triplet.triplet_id} variant="outlined">
            <CardContent>
              <Stack spacing={1.5}>
                <Grid container justifyContent="space-between" alignItems="center">
                  <Grid item>
                    <Typography variant="subtitle1" fontWeight={600}>
                      {head.text || '—'} — {relationValue || '…'} → {tail.text || '—'}
                    </Typography>
                  </Grid>
                  <Grid item>
                    <Stack direction="row" spacing={1}>
                      <Chip
                        size="small"
                        label={`Audit: ${auditStatus}`}
                        color={auditStatus === 'approve' ? 'success' : auditStatus === 'reject' ? 'error' : 'warning'}
                        variant="outlined"
                      />
                      <Chip
                        size="small"
                        label={triplet.rule_support ? 'Rule-backed' : 'LLM-only'}
                        color={ruleChipColor}
                        variant={triplet.rule_support ? 'outlined' : 'default'}
                      />
                      {currentAction !== 'pending' && (
                        <Chip size="small" label={`Action: ${currentAction}`} color="primary" />
                      )}
                    </Stack>
                  </Grid>
                </Grid>

                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <TextField
                      label="Head Entity"
                      fullWidth
                      value={head.text || ''}
                      onChange={(event) =>
                        onUpdate(triplet.triplet_id, {
                          edited: {
                            head: { ...head, text: event.target.value },
                          },
                        })
                      }
                      helperText={head.label || 'Head type'}
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth>
                      <InputLabel>Relation</InputLabel>
                      <Select
                        label="Relation"
                        value={relationValue}
                        onChange={(event) =>
                          onUpdate(triplet.triplet_id, {
                            edited: { relation: event.target.value },
                          })
                        }
                      >
                        {relationOptions.map((relation) => (
                          <MenuItem key={relation} value={relation}>
                            {relation}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      label="Tail Entity"
                      fullWidth
                      value={tail.text || ''}
                      onChange={(event) =>
                        onUpdate(triplet.triplet_id, {
                          edited: {
                            tail: { ...tail, text: event.target.value },
                          },
                        })
                      }
                      helperText={tail.label || 'Tail type'}
                    />
                  </Grid>
                </Grid>

                <TextField
                  label="Evidence"
                  fullWidth
                  multiline
                  minRows={2}
                  value={evidence}
                  onChange={(event) =>
                    onUpdate(triplet.triplet_id, {
                      edited: { evidence: event.target.value },
                    })
                  }
                />

                {triplet.audit?.issues && triplet.audit.issues.length > 0 && (
                  <Stack spacing={0.5}>
                    <Typography variant="caption" color="text.secondary">
                      Auditor flags:
                    </Typography>
                    {triplet.audit.issues.map((issue, index) => (
                      <Typography key={index} variant="body2" color="warning.main">
                        • {issue}
                      </Typography>
                    ))}
                  </Stack>
                )}

                <ButtonGroup fullWidth variant="outlined">
                  {actionButtons.map((action) => (
                    <Tooltip key={action.value} title={`Mark as ${action.label}`} arrow>
                      <Button
                        color={action.color}
                        variant={currentAction === action.value ? 'contained' : 'outlined'}
                        startIcon={action.icon}
                        onClick={() =>
                          onUpdate(triplet.triplet_id, {
                            reviewer_action: action.value,
                          })
                        }
                      >
                        {action.label}
                      </Button>
                    </Tooltip>
                  ))}
                </ButtonGroup>
              </Stack>
            </CardContent>
          </Card>
        );
      })}
    </Stack>
  );
}

TripletReviewPanel.propTypes = {
  triplets: PropTypes.arrayOf(PropTypes.object),
  relationTypes: PropTypes.arrayOf(PropTypes.string),
  onUpdate: PropTypes.func.isRequired,
};

TripletReviewPanel.defaultProps = {
  triplets: [],
  relationTypes: [],
};

export default TripletReviewPanel;

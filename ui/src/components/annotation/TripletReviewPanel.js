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
  Tooltip,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import EditIcon from '@mui/icons-material/Edit';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import Autocomplete from '@mui/material/Autocomplete';

const actionButtons = [
  { value: 'approve', label: 'Approve', icon: <CheckIcon fontSize="small" />, color: 'success' },
  { value: 'reject', label: 'Reject', icon: <CloseIcon fontSize="small" />, color: 'error' },
  { value: 'revise', label: 'Needs Edit', icon: <EditIcon fontSize="small" />, color: 'warning' },
  { value: 'skip', label: 'Skip', icon: <SkipNextIcon fontSize="small" />, color: 'primary' },
];

const normalizeRelationLabel = (relation) =>
  (relation || '')
    .toString()
    .trim()
    .replace(/\s+/g, '_')
    .toUpperCase();

const extractRelationSuggestions = (triplet) => {
  const suggestions = new Set();

  const addSuggestion = (value) => {
    const normalized = normalizeRelationLabel(value);
    if (normalized) {
      suggestions.add(normalized);
    }
  };

  addSuggestion(triplet?.relation);
  addSuggestion(triplet?.edited?.relation);
  addSuggestion(triplet?.audit?.suggested_relation);

  const rawTriplet = triplet?.raw_triplet || {};
  if (Array.isArray(rawTriplet.suggested_relations)) {
    rawTriplet.suggested_relations.forEach(addSuggestion);
  }
  if (Array.isArray(rawTriplet.alternative_relations)) {
    rawTriplet.alternative_relations.forEach(addSuggestion);
  }
  if (rawTriplet.suggested_relation) {
    addSuggestion(rawTriplet.suggested_relation);
  }
  if (rawTriplet.relation_options) {
    const options = Array.isArray(rawTriplet.relation_options)
      ? rawTriplet.relation_options
      : [rawTriplet.relation_options];
    options.forEach(addSuggestion);
  }

  return Array.from(suggestions);
};

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

  return (
    <Stack spacing={2} sx={{ overflowY: 'auto', pr: 1 }}>
      {triplets.map((triplet) => {
        const currentAction = triplet.reviewer_action || 'pending';
        const auditStatus = triplet.audit?.status || 'n/a';
        const ruleChipColor = triplet.rule_support ? 'success' : 'default';
        const head = triplet.edited?.head || triplet.head || {};
        const tail = triplet.edited?.tail || triplet.tail || {};
        const relationValue = normalizeRelationLabel(
          triplet.edited?.relation || triplet.relation || '',
        );
        const evidence = triplet.edited?.evidence ?? triplet.evidence ?? '';

        const relationSuggestions = extractRelationSuggestions(triplet);
        const hasSuggestions = relationSuggestions.length > 0;
        const suggestionsSet = new Set(relationSuggestions);
        const ontologyOptions = hasSuggestions
          ? relationTypes
              .map((relation) => normalizeRelationLabel(relation))
              .filter((relation) => relation && !suggestionsSet.has(relation))
          : [];

        const optionObjects = hasSuggestions
          ? [
              ...relationSuggestions.map((relation) => ({
                label: relation,
                value: relation,
                group: 'LLM Suggestions',
              })),
              ...ontologyOptions.map((relation) => ({
                label: relation,
                value: relation,
                group: 'Ontology',
              })),
            ]
          : [];

        const selectedOption = optionObjects.find((option) => option.value === relationValue);
        const autocompleteValue = relationValue
          ? selectedOption || {
              label: relationValue,
              value: relationValue,
              group: 'Custom',
            }
          : null;

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
                    <Autocomplete
                      freeSolo
                      options={optionObjects}
                      value={autocompleteValue}
                      onChange={(event, newValue) => {
                        let nextRelation = '';
                        if (typeof newValue === 'string') {
                          nextRelation = normalizeRelationLabel(newValue);
                        } else if (newValue && typeof newValue === 'object') {
                          nextRelation = normalizeRelationLabel(newValue.value || newValue.label);
                        }
                        onUpdate(triplet.triplet_id, {
                          edited: { relation: nextRelation },
                        });
                      }}
                      groupBy={(option) => option.group || 'Custom'}
                      isOptionEqualToValue={(option, value) => {
                        const optionValue = typeof option === 'string' ? option : option?.value;
                        const compareValue = typeof value === 'string' ? value : value?.value;
                        return optionValue === compareValue;
                      }}
                      getOptionLabel={(option) => {
                        if (typeof option === 'string') {
                          return option;
                        }
                        return option?.label || '';
                      }}
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label="Relation"
                          placeholder={
                            hasSuggestions
                              ? 'Select or type relation'
                              : 'Type relation manually'
                          }
                        />
                      )}
                    />
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

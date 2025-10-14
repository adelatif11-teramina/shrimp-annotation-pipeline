export const ENTITY_TYPES = Object.freeze({
  // Core biological entities
  SPECIES: { color: '#FF6B6B', description: 'Organism names (shrimp species, bacteria)' },
  PATHOGEN: { color: '#FF8E53', description: 'Disease agents (Vibrio, viruses, parasites)' },
  DISEASE: { color: '#FFD93D', description: 'Clinical conditions/syndromes (AHPND, WSD, TPD)' },
  CLINICAL_SYMPTOM: { color: '#FF4444', description: 'Observed abnormalities (white feces, lethargy)' },
  PHENOTYPIC_TRAIT: { color: '#E91E63', description: 'Measurable performance (survival rate, growth rate)' },
  GENE: { color: '#9B59B6', description: 'Genetic markers (PvIGF, hemocyanin, TLR)' },
  CHEMICAL_COMPOUND: { color: '#FF9800', description: 'Chemical substances (florfenicol, vitamin C, lime)' },
  TREATMENT: { color: '#6BCF7F', description: 'Treatment protocols (antibiotic therapy, vaccination)' },
  LIFE_STAGE: { color: '#00BCD4', description: 'Development stages (PL15, juvenile, broodstock)' },

  // Reified entities
  MEASUREMENT: { color: '#795548', description: 'Reified measurements (28°C, 10 mg/kg, 85%)' },
  SAMPLE: { color: '#8BC34A', description: 'Physical samples with IDs (S-001, water sample W123)' },
  TEST_TYPE: { color: '#03A9F4', description: 'Diagnostic methods (PCR, qPCR, histopathology)' },
  TEST_RESULT: { color: '#009688', description: 'Test outcomes (WSSV positive, Ct=22)' },

  // Operational entities
  MANAGEMENT_PRACTICE: { color: '#4D96FF', description: 'Farming practices (biosecurity, water exchange)' },
  ENVIRONMENTAL_PARAM: { color: '#607D8B', description: 'Environmental factors (temperature, salinity, DO)' },
  LOCATION: { color: '#9E9E9E', description: 'Geographic/facility locations (pond, hatchery, Thailand)' },
  EVENT: { color: '#FF9800', description: 'Timestamped occurrences (mortality event, outbreak)' },
  TISSUE: { color: '#E1BEE7', description: 'Anatomical parts (hepatopancreas, gill, hemolymph)' },

  // Supply chain entities
  PRODUCT: { color: '#FFC107', description: 'Commercial products (PL batch, frozen shrimp)' },
  SUPPLY_ENTITY: { color: '#CDDC39', description: 'Supply chain participants (hatchery, feed supplier)' },
  PERSON: { color: '#F48FB1', description: 'Individuals (farmer, technician, veterinarian)' },
  ORGANIZATION: { color: '#CE93D8', description: 'Companies/institutions (CP Foods, research institute)' },

  // Procedural entities
  PROTOCOL: { color: '#90CAF9', description: 'SOPs (biosecurity protocol, treatment regimen)' },
  CERTIFICATION: { color: '#A5D6A7', description: 'Quality certs (BAP, ASC, organic)' },
});

export const RELATION_TYPES = Object.freeze([
  // Co-reference types (for duplicate entities)
  'same_as',
  'refers_to',
  'abbreviation_of',
  'synonym_of',
  'alias_of',

  // Core biological relations (v2.0)
  'infected_by',
  'infects',
  'causes',
  'treated_with',
  'confers_resistance_to',
  'resistant_to',

  // Risk and protective factors (NEW in v2.0)
  'increases_risk_of',
  'reduces_risk_of',

  // Genetic and physiological (v2.0)
  'expressed_in',
  'affects_trait',
  'has_variant',

  // Sampling and testing (NEW in v2.0)
  'sample_taken_from',
  'tested_with',
  'has_test_result',
  'measurement_of',

  // Chemical-specific relations (NEW with CHEMICAL_COMPOUND)
  'inhibits',
  'enhances',

  // Operational (NEW in v2.0)
  'applied_at',
  'located_in',
  'supplied_by',
  'sold_to',
  'uses_protocol',
  'certified_by',
  'part_of',
]);

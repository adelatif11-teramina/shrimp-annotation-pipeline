# Shrimp Aquaculture Annotation Guidelines

Version 1.0 | Last Updated: January 2025

## Table of Contents
1. [Introduction](#introduction)
2. [Entity Annotation](#entity-annotation)
3. [Relation Annotation](#relation-annotation)
4. [Topic Classification](#topic-classification)
5. [Quality Standards](#quality-standards)
6. [Common Challenges](#common-challenges)
7. [Examples](#examples)

## Introduction

This guide provides comprehensive instructions for annotating shrimp aquaculture scientific literature. Your annotations will be used to train machine learning models and build a knowledge graph for the domain.

### Core Principles
- **Accuracy over Speed**: Take time to make correct annotations
- **When in Doubt, Flag**: Mark uncertain cases for review
- **Evidence-Based**: Only annotate what is explicitly stated
- **Consistency**: Follow guidelines uniformly across all documents

## Entity Annotation

### Entity Types and Definitions

#### SPECIES
**Definition**: Names of organisms, including shrimp species, bacteria, and other biological entities.

**Include**:
- Scientific names: *Penaeus vannamei*, *Litopenaeus vannamei*
- Common names: Pacific white shrimp, tiger shrimp
- Bacterial species: *Vibrio parahaemolyticus*
- General terms when referring to specific species: "the shrimp" (if context is clear)

**Exclude**:
- Generic references: "marine organisms", "aquatic species"
- Non-specific plural forms: "shrimps in general"

**Canonical Forms**:
- Always use full scientific name when available
- *P. vannamei* → *Penaeus vannamei*
- White shrimp → *Penaeus vannamei*

#### PATHOGEN
**Definition**: Disease-causing agents including bacteria, viruses, parasites.

**Include**:
- Bacterial pathogens: *Vibrio* spp., *Aeromonas*
- Viruses: WSSV, YHV, IHHNV
- Parasites: EHP (*Enterocytozoon hepatopenaei*)
- Pathogen strains: VpTPD, VpAHPND

**Exclude**:
- Non-pathogenic bacteria (probiotics)
- Beneficial microorganisms

**Special Cases**:
- Annotate specific strains separately from species
- Include plasmid designations when mentioned

#### DISEASE
**Definition**: Named diseases, syndromes, or clinical conditions.

**Include**:
- Acronyms: AHPND, WSD, TPD, EMS
- Full names: Acute Hepatopancreatic Necrosis Disease
- Syndrome descriptions: white feces syndrome
- Clinical signs when used as disease names

**Exclude**:
- Symptoms alone: "mortality", "slow growth"
- General health states: "poor health", "stress"

**Canonical Forms**:
- EMS = AHPND (use AHPND)
- White spot = WSD

#### TREATMENT
**Definition**: Therapeutic interventions, medications, and management actions.

**Include**:
- Antibiotics: oxytetracycline, florfenicol
- Probiotics: *Bacillus subtilis*, commercial products
- Procedures: vaccination, bacteriophage therapy
- Feed additives: organic acids, essential oils

**Exclude**:
- Preventive measures (use MANAGEMENT_PRACTICE)
- Environmental adjustments

#### GENE
**Definition**: Genetic markers, genes, and genetic variants.

**Include**:
- Gene names: PvIGF, hemocyanin
- Genetic markers: SNP_rs12345
- Gene families: antimicrobial peptides

**Format**:
- Maintain original capitalization
- Include full designation

#### TRAIT
**Definition**: Measurable biological characteristics.

**Include**:
- Performance traits: growth rate, survival rate, FCR
- Resistance traits: disease resistance, pathogen resistance
- Physiological traits: immune response, stress tolerance

**Exclude**:
- Environmental measurements
- Management metrics

#### LIFE_STAGE
**Definition**: Developmental stages of shrimp.

**Include**:
- Larval stages: nauplius, zoea, mysis
- Post-larvae: PL5, PL10, PL15
- Adult stages: juvenile, sub-adult, broodstock

**Format**:
- PL + number indicates days post-metamorphosis
- Always annotate the full designation

#### MEASUREMENT
**Definition**: Numeric values with units.

**Include**:
- Concentrations: 30 ppt, 5 mg/L
- Temperatures: 28°C, 301K
- Weights: 20 g, 15-25 g
- Percentages: 85% survival

**Boundaries**:
- Include the entire measurement with units
- Include ranges completely

#### ENVIRONMENTAL_PARAM
**Definition**: Water quality and environmental factors.

**Include**:
- Parameters: temperature, salinity, pH, DO
- Conditions: alkalinity, ammonia, nitrite

**Exclude**:
- Measurement values (annotate separately as MEASUREMENT)

#### LOCATION
**Definition**: Geographic or facility locations.

**Include**:
- Facility types: pond, hatchery, raceway
- Geographic locations: Thailand, Mekong Delta
- Specific designations: Pond A, Tank 3

## Relation Annotation

### Relation Types

#### infected_by
- **Domain**: SPECIES
- **Range**: PATHOGEN
- **Example**: "*Penaeus vannamei* infected_by *Vibrio parahaemolyticus*"
- **Evidence Required**: Explicit mention of infection

#### causes
- **Domain**: PATHOGEN
- **Range**: DISEASE
- **Example**: "*Vibrio parahaemolyticus* causes AHPND"
- **Evidence Required**: Causal relationship stated

#### treated_with
- **Domain**: DISEASE or SPECIES
- **Range**: TREATMENT
- **Example**: "AHPND treated_with florfenicol"
- **Evidence Required**: Treatment application mentioned

#### resistant_to
- **Domain**: SPECIES, GENE, or LINE
- **Range**: PATHOGEN or DISEASE
- **Example**: "SPR line resistant_to WSSV"
- **Evidence Required**: Resistance explicitly stated

#### measurement_of
- **Domain**: MEASUREMENT
- **Range**: TRAIT or ENVIRONMENTAL_PARAM
- **Example**: "28°C measurement_of temperature"

### Relation Guidelines

1. **Only annotate explicit relationships** - Don't infer unstated connections
2. **Directionality matters** - Head and tail must be correct
3. **Evidence span** - Mark the text supporting the relation
4. **Cross-sentence relations** - Only if pronouns/references are clear
5. **Negative relations** - Mark polarity as "negative"

## Topic Classification

Assign 1-3 most relevant topics per document segment:

### Topic Definitions

- **T_DISEASE**: Disease mechanisms, pathology, outbreaks
- **T_TREATMENT**: Therapeutic interventions, drug trials
- **T_GENETICS**: Breeding, selection, genetic markers
- **T_MANAGEMENT**: Farm operations, husbandry practices
- **T_ENVIRONMENT**: Water quality, environmental monitoring
- **T_BIOSECURITY**: Prevention, quarantine, disinfection
- **T_NUTRITION**: Feed, nutritional requirements
- **T_DIAGNOSTICS**: Detection methods, diagnostic tools

### Topic Assignment Rules

1. **Primary topic** - The main focus (highest confidence)
2. **Secondary topics** - Supporting themes (medium confidence)
3. **Justify with keywords** - List 3-5 keywords supporting assignment
4. **Document-level coherence** - Consider full document context

## Quality Standards

### Confidence Levels

- **High**: Certain about annotation (>90% confident)
- **Medium**: Likely correct but some uncertainty (70-90%)
- **Low**: Uncertain, needs review (<70%)

### When to Flag for Review

Flag annotations when:
- Entity boundaries are unclear
- Multiple valid interpretations exist
- Technical terms are unfamiliar
- Abbreviations are ambiguous
- Context is insufficient

### Inter-Annotator Agreement

For double-annotated documents:
- Discuss disagreements with lead annotator
- Document resolution rationale
- Update guidelines based on edge cases

## Common Challenges

### Abbreviation Resolution

**Challenge**: Unknown abbreviations
**Solution**: 
1. Check document for first use/definition
2. Consult domain glossary
3. Flag if unresolvable

### Boundary Detection

**Challenge**: Determining entity span
**Examples**:
- ✓ "Vibrio parahaemolyticus AHPND strain" - Annotate as one PATHOGEN
- ✗ "Vibrio" and "parahaemolyticus" separately
- ✓ "15-20 g" - Full range as MEASUREMENT
- ✗ "15" and "20 g" separately

### Nested Entities

**Challenge**: Entities within entities
**Solution**: Annotate the most specific appropriate type
- "AHPND-causing Vibrio" → PATHOGEN for full span
- Can annotate "AHPND" separately if discussed independently

### Ambiguous Sentences

**Challenge**: Multiple valid interpretations
**Example**: "The treatment improved survival"
**Solutions**:
1. Check previous sentences for treatment identity
2. If clear from context, annotate
3. If ambiguous, skip relation annotation

## Examples

### Example 1: Basic Annotation

**Text**: "Vibrio parahaemolyticus causes AHPND in Penaeus vannamei post-larvae at 30 ppt salinity."

**Entities**:
- "Vibrio parahaemolyticus" - PATHOGEN
- "AHPND" - DISEASE
- "Penaeus vannamei" - SPECIES
- "post-larvae" - LIFE_STAGE
- "30 ppt" - MEASUREMENT
- "salinity" - ENVIRONMENTAL_PARAM

**Relations**:
- Vibrio parahaemolyticus **causes** AHPND
- Penaeus vannamei **infected_by** Vibrio parahaemolyticus
- 30 ppt **measurement_of** salinity

### Example 2: Complex Treatment

**Text**: "Florfenicol at 10 mg/kg feed for 7 days reduced mortality from AHPND by 60% in challenged shrimp."

**Entities**:
- "Florfenicol" - TREATMENT
- "10 mg/kg" - MEASUREMENT
- "7 days" - MEASUREMENT
- "mortality" - TRAIT
- "AHPND" - DISEASE
- "60%" - MEASUREMENT
- "shrimp" - SPECIES

**Relations**:
- AHPND **treated_with** Florfenicol
- 60% **measurement_of** mortality

### Example 3: Genetic Resistance

**Text**: "The PvIGF gene variant was significantly associated with WSSV resistance in the SPR breeding line."

**Entities**:
- "PvIGF" - GENE
- "WSSV" - PATHOGEN
- "resistance" - TRAIT
- "SPR breeding line" - SPECIES

**Relations**:
- PvIGF **affects_trait** resistance
- SPR breeding line **resistant_to** WSSV

## Edge Cases and Decisions

### Case 1: Implied vs Explicit
**Text**: "Infected shrimp showed high mortality"
**Decision**: Don't create relation without explicit pathogen mention

### Case 2: Speculation
**Text**: "This might be caused by Vibrio"
**Decision**: Mark relation with polarity="speculative"

### Case 3: Lists
**Text**: "Vibrio, Aeromonas, and Pseudomonas were isolated"
**Decision**: Annotate each species separately

## Revision History

- v1.0 (Jan 2025): Initial guidelines
- Updates will be logged here with rationale

## Contact

For questions or clarifications:
- Lead Annotator: [Email]
- Project Manager: [Email]
- Guidelines Discussion: [Slack/Teams Channel]
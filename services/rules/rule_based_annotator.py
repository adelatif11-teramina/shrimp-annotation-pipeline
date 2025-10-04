"""
Rule-based Annotation Service

Integrates rule-based pattern matching from data-training project
with the new annotation pipeline for disagreement detection and baseline annotations.
Updated to use v2.0 ontology with reduced noise patterns.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys

# Add data-training project to path for importing
data_training_path = Path(__file__).parent.parent.parent.parent / "data-training"
sys.path.append(str(data_training_path))

logger = logging.getLogger(__name__)

@dataclass
class RuleEntity:
    """Rule-based entity result"""
    cid: int
    text: str
    label: str
    start: int
    end: int
    confidence: float
    rule_type: str  # scientific, common, pattern, etc.
    rule_source: str  # which rule matched

@dataclass
class RuleRelation:
    """Rule-based relation result"""
    rid: int
    head_cid: int
    tail_cid: int
    label: str
    evidence: str
    confidence: float
    rule_pattern: str

class ShimpAquacultureRuleEngine:
    """
    Rule-based annotation engine for shrimp aquaculture domain.
    
    Adapts patterns from data-training project for use in the annotation pipeline.
    Updated for v2.0 ontology with improved precision.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize rule engine.
        
        Args:
            config_path: Path to rule configuration file
        """
        self.config_path = config_path
        self._load_patterns()
        self._load_ontology()
        
        # Statistics
        self.stats = {
            "entities_matched": 0,
            "relations_matched": 0,
            "rules_fired": defaultdict(int)
        }
        
        logger.info("Initialized rule-based annotation engine v2.0")
    
    def _load_patterns(self):
        """Load annotation patterns for v2.0 ontology"""
        
        # SPECIES patterns - specific shrimp species only (high precision)
        self.species_patterns = [
            # Scientific names (high confidence)
            (r'\bPenaeus\s+(?:vannamei|monodon|japonicus|indicus|stylirostris)\b', 'scientific', 0.95),
            (r'\bLitopenaeus\s+(?:vannamei|stylirostris|setiferus)\b', 'scientific', 0.95),
            (r'\bFenneropenaeus\s+(?:chinensis|indicus|merguiensis)\b', 'scientific', 0.95),
            # Common names (medium-high confidence)
            (r'\b(?:Pacific\s+)?white(?:[-\s]*leg)?\s+shrimp\b', 'common', 0.9),
            (r'\b(?:black\s+)?tiger\s+(?:shrimp|prawn)\b', 'common', 0.9),
            (r'\bblue\s+shrimp\b', 'common', 0.85),
            # General terms only in specific contexts (lower confidence)
            (r'\b(?:post[-\s]*larvae?|juvenile|broodstock)\s+shrimp\b', 'contextual', 0.7),
        ]
        
        # PATHOGEN patterns (v2.0)
        self.pathogen_patterns = [
            # Vibrio species (high confidence)
            (r'\bVibrio\s+parahaemolyticus\b', 'bacteria', 0.95),
            (r'\bVibrio\s+harveyi\b', 'bacteria', 0.95),
            (r'\bVibrio\s+(?:vulnificus|alginolyticus|campbellii|owensii)\b', 'bacteria', 0.9),
            (r'\bV\.\s+parahaemolyticus\b', 'bacteria_abbrev', 0.9),
            # Other bacterial pathogens
            (r'\bAeromonas\s+(?:hydrophila|sobria|caviae)\b', 'bacteria', 0.9),
            (r'\bPhotobacterium\s+damselae\b', 'bacteria', 0.9),
            # Microsporidians
            (r'\b(?:EHP|Enterocytozoon\s+hepatopenaei)\b', 'microsporidian', 0.95),
            # Viruses
            (r'\bWhite\s+[Ss]pot\s+[Ss]yndrome\s+[Vv]irus\b', 'virus', 0.95),
            (r'\bTaura\s+[Ss]yndrome\s+[Vv]irus\b', 'virus', 0.95),
            (r'\bYellow\s+[Hh]ead\s+[Vv]irus\b', 'virus', 0.95),
            (r'\b(?:WSSV|TSV|YHV|IHHNV|IMNV)\b', 'virus_acronym', 0.95),
        ]
        
        # DISEASE patterns (v2.0)
        self.disease_patterns = [
            # Specific diseases (high confidence)
            (r'\b(?:AHPND|Acute\s+Hepatopancreatic\s+Necrosis\s+Disease)\b', 'specific', 0.95),
            (r'\b(?:TPD|[Tt]ranslucent\s+[Pp]ost[-\s]*larvae?\s+[Dd]isease)\b', 'specific', 0.95),
            (r'\b(?:WSD|[Ww]hite\s+[Ss]pot\s+[Dd]isease)\b', 'specific', 0.95),
            (r'\b(?:EMS|[Ee]arly\s+[Mm]ortality\s+[Ss]yndrome)\b', 'specific', 0.9),
            (r'\bTaura\s+syndrome\b', 'specific', 0.9),
            (r'\bWhite\s+feces\s+syndrome\b', 'specific', 0.9),
            (r'\bRunning\s+mortality\s+syndrome\b', 'specific', 0.85),
            (r'\bHepatopancreatic\s+microsporidiosis\b', 'specific', 0.85),
        ]
        
        # CLINICAL_SYMPTOM patterns (NEW in v2.0)
        self.clinical_symptom_patterns = [
            (r'\bwhite\s+feces\b', 'digestive', 0.9),
            (r'\blethargy\b', 'behavioral', 0.85),
            (r'\babnormal\s+swimming\b', 'behavioral', 0.85),
            (r'\bgrowth\s+retardation\b', 'growth', 0.85),
            (r'\bsoft\s+shell\b', 'structural', 0.85),
            (r'\bmuscle\s+necrosis\b', 'tissue', 0.9),
            (r'\bmelanization\b', 'coloration', 0.85),
            (r'\bred\s+discoloration\b', 'coloration', 0.85),
            (r'\bempty\s+gut\b', 'digestive', 0.85),
            (r'\bwhite\s+muscle\b', 'tissue', 0.85),
            (r'\bgill\s+damage\b', 'tissue', 0.85),
        ]
        
        # PHENOTYPIC_TRAIT patterns (NEW in v2.0)
        self.phenotypic_trait_patterns = [
            (r'\bsurvival\s+rate\b', 'performance', 0.9),
            (r'\bgrowth\s+rate\b', 'performance', 0.9),
            (r'\bdisease\s+resistance\b', 'resistance', 0.85),
            (r'\bfeed\s+conversion\s+ratio\b', 'efficiency', 0.9),
            (r'\b(?:FCR|ADG)\b', 'efficiency_abbrev', 0.85),
            (r'\bspecific\s+pathogen\s+resistance\b', 'resistance', 0.9),
            (r'\baverage\s+daily\s+gain\b', 'performance', 0.85),
            (r'\bbiomass\b', 'production', 0.8),
            (r'\byield\b', 'production', 0.75),
        ]
        
        # TREATMENT patterns (v2.0 enhanced)
        self.treatment_patterns = [
            # Specific antibiotics
            (r'\b(?:oxytetracycline|florfenicol|enrofloxacin|sulfadiazine)\b', 'antibiotic', 0.95),
            # Probiotic patterns
            (r'\bBacillus\s+(?:subtilis|licheniformis|pumilus|coagulans)\b', 'probiotic', 0.9),
            (r'\bLactobacillus\s+(?:plantarum|acidophilus|rhamnosus)\b', 'probiotic', 0.9),
            (r'\bSaccharomyces\s+(?:cerevisiae|boulardii)\b', 'probiotic', 0.85),
            # Immunostimulants
            (r'\bbeta[-\s]*glucan\b', 'immunostimulant', 0.85),
            (r'\bpeptidoglycan\b', 'immunostimulant', 0.85),
            # Disinfectants
            (r'\b(?:chlorine|iodine|formalin|potassium\s+permanganate)\b', 'disinfectant', 0.85),
            # General treatments
            (r'\bprobiotic\s+treatment\b', 'biological', 0.8),
            (r'\bantibiotic\s+therapy\b', 'chemical', 0.8),
        ]
        
        # MEASUREMENT patterns (v2.0 reified)
        self.measurement_patterns = [
            # Temperature
            (r'\b(\d+\.?\d*)\s*(°C|degrees?\s+C(?:elsius)?)\b', 'temperature', 0.95),
            # Salinity
            (r'\b(\d+\.?\d*)\s*(ppt|g/L|‰|parts?\s+per\s+thousand)\b', 'salinity', 0.95),
            # Weight/size
            (r'\b(\d+\.?\d*)\s*(g|kg|mg|grams?|kilograms?)\b', 'weight', 0.9),
            # Percentage
            (r'\b(\d+\.?\d*)\s*(%|percent)\b', 'percentage', 0.9),
            # Concentration
            (r'\b(\d+\.?\d*)\s*(mg/L|μg/L|ppm|ppb)\b', 'concentration', 0.95),
            # pH
            (r'\bpH\s*(\d+\.?\d*)\b', 'pH', 0.95),
            # Dissolved oxygen
            (r'\b(\d+\.?\d*)\s*mg/L\s+DO\b', 'dissolved_oxygen', 0.9),
            # Stocking density
            (r'\b(\d+)\s*(?:shrimp|PL|post[-\s]*larvae?)/m[²2]\b', 'density', 0.9),
        ]
        
        # LIFE_STAGE patterns (v2.0 enhanced)
        self.life_stage_patterns = [
            (r'\bPL(?:\s*)?(\d+)\b', 'post_larvae_day', 0.95),
            (r'\bpost[-\s]*larvae?\b', 'post_larvae', 0.9),
            (r'\bjuvenile\b', 'juvenile', 0.9),
            (r'\bsub[-\s]*adult\b', 'sub_adult', 0.85),
            (r'\bbroodstock\b', 'broodstock', 0.9),
            (r'\bnauplius\b', 'nauplius', 0.95),
            (r'\bzoea\b', 'zoea', 0.95),
            (r'\bmysis\b', 'mysis', 0.95),
        ]
        
        # GENE patterns (v2.0)
        self.gene_patterns = [
            (r'\bPvIGF\b', 'growth', 0.9),
            (r'\bhemocyanin\b', 'immunity', 0.85),
            (r'\bprophenoloxidase\b', 'immunity', 0.85),
            (r'\bcrustin\b', 'antimicrobial', 0.85),
            (r'\bpenaeidin\b', 'antimicrobial', 0.85),
            (r'\blysozyme\b', 'antimicrobial', 0.85),
            (r'\btoll[-\s]*like\s+receptor\b', 'immunity', 0.85),
            (r'\bTLR\b', 'immunity_abbrev', 0.8),
            (r'\bvhvp[-_]?[12]\b', 'virulence', 0.9),
        ]
        
        # SAMPLE patterns (NEW in v2.0)
        self.sample_patterns = [
            (r'\b[Ss]ample\s+[A-Z0-9][-0-9]+\b', 'sample_id', 0.9),
            (r'\b[A-Z]{1,3}[-]\d{3,}\b', 'sample_code', 0.8),
            (r'\bwater\s+sample\s+[A-Z0-9]+\b', 'water_sample', 0.9),
            (r'\bhepatopancreas\s+sample\b', 'tissue_sample', 0.85),
            (r'\bhemolymph\s+sample\b', 'tissue_sample', 0.85),
        ]
        
        # TEST_TYPE patterns (NEW in v2.0)
        self.test_type_patterns = [
            (r'\bPCR\b', 'molecular', 0.95),
            (r'\bq(?:uantitative[-\s]*)?PCR\b', 'molecular', 0.95),
            (r'\bRT[-\s]*PCR\b', 'molecular', 0.95),
            (r'\bLAMP\b', 'molecular', 0.9),
            (r'\bhistopathology\b', 'microscopy', 0.9),
            (r'\bbacterial\s+culture\b', 'culture', 0.9),
            (r'\bELISA\b', 'immunoassay', 0.95),
            (r'\bimmunofluorescence\b', 'immunoassay', 0.9),
            (r'\bbioassay\b', 'bioassay', 0.85),
        ]
        
        # TEST_RESULT patterns (NEW in v2.0)
        self.test_result_patterns = [
            (r'\b(?:WSSV|TSV|YHV|EHP)\s+positive\b', 'pathogen_positive', 0.95),
            (r'\b(?:WSSV|TSV|YHV|EHP)\s+negative\b', 'pathogen_negative', 0.95),
            (r'\bCt\s*(?:value\s*)?(?:=|:)?\s*(\d+\.?\d*)\b', 'ct_value', 0.9),
            (r'\b(\d+\.?\d*)\s*(?:×|x)\s*10\^(\d+)\s*CFU/mL\b', 'bacterial_count', 0.9),
            (r'\bpositive\s+for\s+\w+\b', 'positive_result', 0.85),
            (r'\bnegative\s+for\s+\w+\b', 'negative_result', 0.85),
        ]
        
        # MANAGEMENT_PRACTICE patterns (v2.0 enhanced)
        self.management_practice_patterns = [
            (r'\bbiosecurity\b', 'biosecurity', 0.9),
            (r'\bwater\s+exchange\b', 'water_mgmt', 0.9),
            (r'\bpond\s+preparation\b', 'pond_mgmt', 0.85),
            (r'\bstocking\s+density\b', 'stocking', 0.9),
            (r'\bbiofloc\s+(?:system|technology)\b', 'biofloc', 0.95),
            (r'\bpartial\s+harvest(?:ing)?\b', 'harvest', 0.85),
            (r'\bquarantine\b', 'biosecurity', 0.85),
            (r'\bdisinfection\b', 'biosecurity', 0.85),
            (r'\baeration\b', 'water_mgmt', 0.85),
            (r'\bpH\s+adjustment\b', 'water_mgmt', 0.85),
            (r'\bliming\b', 'pond_mgmt', 0.85),
        ]
    
    def _load_ontology(self):
        """Load domain ontology v2.0 for canonical mapping"""
        ontology_path = Path(__file__).parent.parent.parent / "shared/ontology/shrimp_domain_ontology.yaml"
        
        # Enhanced canonical mappings for v2.0
        self.canonical_mapping = {
            # Species canonicals
            "white shrimp": "Penaeus vannamei",
            "whiteleg shrimp": "Penaeus vannamei", 
            "pacific white shrimp": "Penaeus vannamei",
            "p. vannamei": "Penaeus vannamei",
            "l. vannamei": "Litopenaeus vannamei",
            "tiger shrimp": "Penaeus monodon",
            "black tiger shrimp": "Penaeus monodon",
            "p. monodon": "Penaeus monodon",
            
            # Pathogen canonicals
            "v. parahaemolyticus": "Vibrio parahaemolyticus",
            "v. harveyi": "Vibrio harveyi",
            "white spot virus": "WSSV",
            "white spot syndrome virus": "WSSV",
            "yellow head virus": "YHV",
            "taura syndrome virus": "TSV",
            
            # Disease canonicals
            "ems": "AHPND",
            "early mortality syndrome": "AHPND",
            "white spot": "WSD",
            "white spot disease": "WSD",
            "wfs": "White feces syndrome",
            
            # Treatment canonicals
            "fcr": "feed conversion ratio",
            "adg": "average daily gain",
            
            # Test canonicals
            "real-time pcr": "qPCR",
            "quantitative pcr": "qPCR",
            "ifa": "immunofluorescence",
        }
        
        # Load v2.0 ontology labels
        self.v2_entity_types = {
            "SPECIES", "PATHOGEN", "DISEASE", "CLINICAL_SYMPTOM", "PHENOTYPIC_TRAIT",
            "GENE", "TREATMENT", "MEASUREMENT", "LIFE_STAGE", "MANAGEMENT_PRACTICE",
            "ENVIRONMENTAL_PARAM", "LOCATION", "SAMPLE", "TEST_TYPE", "TEST_RESULT",
            "EVENT", "TISSUE", "PRODUCT", "SUPPLY_ENTITY", "PERSON", "ORGANIZATION",
            "PROTOCOL", "CERTIFICATION"
        }
    
    def extract_entities(self, text: str, sent_start_offset: int = 0) -> List[RuleEntity]:
        """
        Extract entities using rule-based patterns for v2.0 ontology.
        
        Args:
            text: Input text
            sent_start_offset: Offset for adjusting character positions
            
        Returns:
            List of rule entities
        """
        entities = []
        entity_id = 0
        
        # Define pattern groups with v2.0 entity types
        pattern_groups = [
            (self.species_patterns, "SPECIES"),
            (self.pathogen_patterns, "PATHOGEN"), 
            (self.disease_patterns, "DISEASE"),
            (self.clinical_symptom_patterns, "CLINICAL_SYMPTOM"),
            (self.phenotypic_trait_patterns, "PHENOTYPIC_TRAIT"),
            (self.treatment_patterns, "TREATMENT"),
            (self.measurement_patterns, "MEASUREMENT"),
            (self.life_stage_patterns, "LIFE_STAGE"),
            (self.gene_patterns, "GENE"),
            (self.sample_patterns, "SAMPLE"),
            (self.test_type_patterns, "TEST_TYPE"),
            (self.test_result_patterns, "TEST_RESULT"),
            (self.management_practice_patterns, "MANAGEMENT_PRACTICE"),
        ]
        
        # Track matched spans to avoid overlaps
        matched_spans = []
        
        for patterns, entity_type in pattern_groups:
            for pattern, rule_type, confidence in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = match.start() + sent_start_offset
                    end = match.end() + sent_start_offset
                    matched_text = match.group().strip()
                    
                    # Check for overlaps
                    overlaps = any(
                        not (end <= span_start or start >= span_end)
                        for span_start, span_end in matched_spans
                    )
                    
                    if not overlaps and matched_text:
                        # Apply canonical mapping
                        canonical = self.canonical_mapping.get(matched_text.lower(), matched_text)
                        
                        entity = RuleEntity(
                            cid=entity_id,
                            text=matched_text,
                            label=entity_type,
                            start=start,
                            end=end,
                            confidence=confidence,
                            rule_type=rule_type,
                            rule_source=pattern
                        )
                        
                        entities.append(entity)
                        matched_spans.append((start, end))
                        entity_id += 1
                        
                        # Update statistics
                        self.stats["entities_matched"] += 1
                        self.stats["rules_fired"][entity_type] += 1
        
        return entities
    
    def extract_relations(self, text: str, entities: List[RuleEntity]) -> List[RuleRelation]:
        """
        Extract relations between entities using v2.0 relation types.
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            List of rule relations
        """
        relations = []
        relation_id = 0
        
        # Define relation patterns for v2.0 ontology
        relation_patterns = [
            # infected_by pattern
            {
                "pattern": r"infected\s+(?:by|with)",
                "head_type": "SPECIES",
                "tail_type": "PATHOGEN", 
                "relation": "infected_by",
                "confidence": 0.9
            },
            # causes pattern
            {
                "pattern": r"caus(?:es?|ed|ing)",
                "head_type": "PATHOGEN",
                "tail_type": ["DISEASE", "CLINICAL_SYMPTOM"],
                "relation": "causes",
                "confidence": 0.85
            },
            # treated_with pattern
            {
                "pattern": r"treat(?:ed|ment)?\s+(?:with|using|by)",
                "head_type": ["DISEASE", "CLINICAL_SYMPTOM"],
                "tail_type": "TREATMENT",
                "relation": "treated_with",
                "confidence": 0.85
            },
            # increases_risk_of pattern (NEW in v2.0)
            {
                "pattern": r"increas(?:es?|ed|ing)\s+(?:the\s+)?risk\s+(?:of|for)",
                "head_type": ["ENVIRONMENTAL_PARAM", "MANAGEMENT_PRACTICE"],
                "tail_type": ["DISEASE", "CLINICAL_SYMPTOM"],
                "relation": "increases_risk_of",
                "confidence": 0.85
            },
            # reduces_risk_of pattern (NEW in v2.0)
            {
                "pattern": r"(?:reduc|decreas)(?:es?|ed|ing)\s+(?:the\s+)?risk\s+(?:of|for)",
                "head_type": ["MANAGEMENT_PRACTICE", "TREATMENT"],
                "tail_type": ["DISEASE", "CLINICAL_SYMPTOM"],
                "relation": "reduces_risk_of",
                "confidence": 0.85
            },
            # tested_with pattern (NEW in v2.0)
            {
                "pattern": r"test(?:ed|ing)?\s+(?:with|using|by)",
                "head_type": "SAMPLE",
                "tail_type": "TEST_TYPE",
                "relation": "tested_with",
                "confidence": 0.9
            },
            # affects_trait pattern (NEW in v2.0)
            {
                "pattern": r"(?:affect|influenc|impact)(?:s|ed|ing)?",
                "head_type": ["GENE", "ENVIRONMENTAL_PARAM", "MANAGEMENT_PRACTICE"],
                "tail_type": "PHENOTYPIC_TRAIT",
                "relation": "affects_trait",
                "confidence": 0.8
            },
        ]
        
        # Look for relations based on patterns
        for rel_pattern in relation_patterns:
            pattern = rel_pattern["pattern"]
            
            # Find pattern matches in text
            for match in re.finditer(pattern, text, re.IGNORECASE):
                evidence = match.group()
                match_pos = match.start()
                
                # Find entities that could be head and tail
                head_candidates = []
                tail_candidates = []
                
                for entity in entities:
                    # Check entity type matches
                    head_types = rel_pattern["head_type"] if isinstance(rel_pattern["head_type"], list) else [rel_pattern["head_type"]]
                    tail_types = rel_pattern["tail_type"] if isinstance(rel_pattern["tail_type"], list) else [rel_pattern["tail_type"]]
                    
                    if entity.label in head_types:
                        # Check if entity is before the pattern
                        if entity.end <= match_pos:
                            distance = match_pos - entity.end
                            head_candidates.append((entity, distance))
                    
                    if entity.label in tail_types:
                        # Check if entity is after the pattern
                        if entity.start >= match_pos + len(evidence):
                            distance = entity.start - (match_pos + len(evidence))
                            tail_candidates.append((entity, distance))
                
                # Find closest head and tail
                if head_candidates and tail_candidates:
                    # Sort by distance
                    head_candidates.sort(key=lambda x: x[1])
                    tail_candidates.sort(key=lambda x: x[1])
                    
                    # Take closest entities within reasonable distance (50 chars)
                    head_entity, head_dist = head_candidates[0]
                    tail_entity, tail_dist = tail_candidates[0]
                    
                    if head_entity is not None and tail_entity is not None and head_entity != tail_entity:
                        if head_dist < 50 and tail_dist < 50:
                            relation = RuleRelation(
                                rid=relation_id,
                                head_cid=head_entity.cid,
                                tail_cid=tail_entity.cid,
                                label=rel_pattern["relation"],
                                evidence=evidence,
                                confidence=rel_pattern["confidence"],
                                rule_pattern=pattern
                            )
                            
                            relations.append(relation)
                            relation_id += 1
                            
                            # Update statistics
                            self.stats["relations_matched"] += 1
                            self.stats["rules_fired"][rel_pattern["relation"]] += 1
        
        return relations
    
    def process_sentence(self, doc_id: str, sent_id: str, text: str) -> Dict[str, Any]:
        """
        Process a sentence to extract entities and relations using v2.0 ontology.
        
        Args:
            doc_id: Document ID
            sent_id: Sentence ID
            text: Sentence text
            
        Returns:
            Dictionary with extraction results
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        
        # Convert to dictionary format
        result = {
            "doc_id": doc_id,
            "sent_id": sent_id,
            "text": text,
            "entities": [asdict(e) for e in entities],
            "relations": [asdict(r) for r in relations],
            "topics": [],  # Rule engine doesn't do topics
            "source": "rule_engine_v2.0",
            "created_at": datetime.now().isoformat() if hasattr(__builtins__, 'datetime') else None
        }
        
        return result


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Test rule-based annotator v2.0")
    parser.add_argument("--text", help="Text to annotate")
    parser.add_argument("--file", help="File to process")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = ShimpAquacultureRuleEngine()
    
    if args.text:
        # Test single sentence
        result = engine.process_sentence("test_doc", "test_sent", args.text)
        print(json.dumps(result, indent=2))
        
    elif args.file:
        # Process file
        with open(args.file, 'r') as f:
            text = f.read()
        
        result = engine.process_sentence("file_doc", "file_sent", text)
        print(json.dumps(result, indent=2))
    
    # Print statistics
    print("\n=== Statistics ===")
    print(json.dumps(engine.stats, indent=2))
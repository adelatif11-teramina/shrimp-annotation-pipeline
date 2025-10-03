"""
Rule-based Annotation Service

Integrates rule-based pattern matching from data-training project
with the new annotation pipeline for disagreement detection and baseline annotations.
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
        
        logger.info("Initialized rule-based annotation engine")
    
    def _load_patterns(self):
        """Load annotation patterns"""
        
        # Species patterns - from data-training automated_annotator.py
        self.species_patterns = [
            # Scientific names
            (r'\b[A-Z][a-z]+ [a-z]+\b', 'scientific', 0.8),
            (r'\bPenaeus\s+(?:vannamei|monodon|japonicus|indicus)\b', 'scientific', 0.95),
            (r'\bLitopenaeus\s+vannamei\b', 'scientific', 0.95),
            # Common names
            (r'\b(?:white[-\s]*leg|whiteleg)\s+shrimp\b', 'common', 0.9),
            (r'\b(?:tiger|black tiger)\s+(?:shrimp|prawn)\b', 'common', 0.9),
            (r'\bPacific\s+white\s+shrimp\b', 'common', 0.95),
            # General terms (lower confidence)
            (r'\b(?:shrimp|prawn|prawns)\b', 'general', 0.6),
        ]
        
        # Pathogen patterns
        self.pathogen_patterns = [
            # Vibrio species (high confidence)
            (r'\bVibrio\s+parahaemolyticus\b', 'bacteria', 0.95),
            (r'\bVibrio\s+harveyi\b', 'bacteria', 0.95),
            (r'\bVibrio\s+(?:vulnificus|alginolyticus|campbellii)\b', 'bacteria', 0.9),
            (r'\bVibrio\s+spp?\.\b', 'bacteria', 0.85),
            # Other pathogens
            (r'\b(?:EHP|Enterocytozoon\s+hepatopenaei)\b', 'microsporidian', 0.95),
            (r'\bWhite\s+spot\s+syndrome\s+virus\b', 'virus', 0.95),
            (r'\bTaura\s+syndrome\s+virus\b', 'virus', 0.95),
            (r'\b(?:WSSV|TSV|YHV|IHHNV)\b', 'virus_acronym', 0.9),
            # General pathogen terms
            (r'\b(?:pathogen|microbe)\b', 'general', 0.7),
        ]
        
        # Disease patterns
        self.disease_patterns = [
            # Specific diseases (high confidence)
            (r'\b(?:AHPND|Acute\s+Hepatopancreatic\s+Necrosis\s+Disease)\b', 'specific', 0.95),
            (r'\b(?:TPD|[Tt]ranslucent\s+[Pp]ost[-\s]*larvae?\s+[Dd]isease)\b', 'specific', 0.95),
            (r'\b(?:WSD|[Ww]hite\s+[Ss]pot\s+[Dd]isease)\b', 'specific', 0.95),
            (r'\b(?:EMS|[Ee]arly\s+[Mm]ortality\s+[Ss]yndrome)\b', 'specific', 0.9),
            (r'\bTaura\s+syndrome\b', 'specific', 0.9),
            (r'\bWhite\s+feces\s+syndrome\b', 'specific', 0.85),
            # General disease terms
            (r'\b(?:syndrome|infection)\b', 'general', 0.6),
        ]
        
        # Treatment patterns
        self.treatment_patterns = [
            # Specific antibiotics
            (r'\b(?:oxytetracycline|florfenicol|enrofloxacin)\b', 'antibiotic', 0.9),
            # Probiotic patterns
            (r'\bBacillus\s+(?:subtilis|licheniformis|pumilus)\b', 'probiotic', 0.85),
            (r'\bLactobacillus\s+\w+\b', 'probiotic', 0.8),
            # General treatments
            (r'\b(?:probiotic|antibiotic|vaccination|immunostimulant)\b', 'biological', 0.8),
            (r'\b(?:disinfection|disinfectant|chlorination)\b', 'chemical', 0.8),
        ]
        
        # Measurement patterns
        self.measurement_patterns = [
            # Temperature
            (r'\b\d+\.?\d*\s*°?C\b', 'temperature', 0.9),
            # Salinity
            (r'\b\d+\.?\d*\s*(?:ppt|g/L|‰)\b', 'salinity', 0.9),
            # Weight/size
            (r'\b\d+\.?\d*\s*(?:g|kg|mg)\b', 'weight', 0.85),
            # Percentage
            (r'\b\d+\.?\d*\s*%\b', 'percentage', 0.85),
            # Concentration
            (r'\b\d+\.?\d*\s*(?:mg/L|μg/L|ppm)\b', 'concentration', 0.9),
        ]
        
        # Life stage patterns
        self.life_stage_patterns = [
            (r'\bPL\d+\b', 'post_larvae', 0.95),
            (r'\b(?:post[-\s]*larvae?|postlarvae?)\b', 'post_larvae', 0.9),
            (r'\b(?:juvenile|sub[-\s]*adult|broodstock)\b', 'development', 0.85),
            (r'\b(?:nauplius|zoea|mysis)\b', 'larval', 0.9),
        ]
        
        # Gene patterns
        self.gene_patterns = [
            (r'\b(?:PvIGF|hemocyanin|prophenoloxidase)\b', 'immunity', 0.8),
            (r'\b(?:crustin|penaeidin|lysozyme)\b', 'antimicrobial', 0.8),
            (r'\bvhvp[-_]?[123]?\b', 'virulence', 0.85),
        ]
    
    def _load_ontology(self):
        """Load domain ontology for canonical mapping"""
        ontology_path = Path(__file__).parent.parent.parent / "shared/ontology/shrimp_domain_ontology.yaml"
        
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
            
            # Disease canonicals
            "ems": "AHPND",
            "early mortality syndrome": "AHPND",
            "white spot": "WSD",
            "white spot disease": "WSD",
        }
    
    def extract_entities(self, text: str, sent_start_offset: int = 0) -> List[RuleEntity]:
        """
        Extract entities using rule-based patterns.
        
        Args:
            text: Input text
            sent_start_offset: Offset for adjusting character positions
            
        Returns:
            List of rule entities
        """
        entities = []
        entity_id = 0
        
        # Define pattern groups
        pattern_groups = [
            (self.species_patterns, "SPECIES"),
            (self.pathogen_patterns, "PATHOGEN"), 
            (self.disease_patterns, "DISEASE"),
            (self.treatment_patterns, "TREATMENT"),
            (self.measurement_patterns, "MEASUREMENT"),
            (self.life_stage_patterns, "LIFE_STAGE"),
            (self.gene_patterns, "GENE"),
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
                        self.stats["rules_fired"][f"{entity_type}_{rule_type}"] += 1
        
        return entities
    
    def extract_relations(self, text: str, entities: List[RuleEntity]) -> List[RuleRelation]:
        """
        Extract relations using rule-based patterns.
        
        Args:
            text: Input text
            entities: List of entities in the text
            
        Returns:
            List of rule relations
        """
        relations = []
        relation_id = 0
        
        # Create entity lookup by position
        entity_positions = [(e.start, e.end, e.cid, e.label, e.text) for e in entities]
        entity_positions.sort()
        
        # Relation patterns
        relation_patterns = [
            # Causal relations
            (r'(\w+(?:\s+\w+)*)\s+(?:causes?|causing|leads?\s+to|results?\s+in)\s+(\w+(?:\s+\w+)*)', 'causes', 0.8),
            (r'(\w+(?:\s+\w+)*)\s+(?:infected?\s+by|infected?\s+with)\s+(\w+(?:\s+\w+)*)', 'infected_by', 0.85),
            
            # Treatment relations  
            (r'(?:treat\w*|therapy|medication)\s+(?:with|using)\s+(\w+(?:\s+\w+)*)', 'treated_with', 0.8),
            (r'(\w+(?:\s+\w+)*)\s+(?:treated?\s+with|therapy\s+with)\s+(\w+(?:\s+\w+)*)', 'treated_with', 0.8),
            
            # Resistance relations
            (r'(\w+(?:\s+\w+)*)\s+(?:resistant\s+to|resistance\s+to)\s+(\w+(?:\s+\w+)*)', 'resistant_to', 0.85),
            
            # Measurement relations
            (r'(\d+\.?\d*\s*(?:°C|ppt|g|%|mg/L))\s+(?:of|for)\s+(\w+)', 'measurement_of', 0.8),
        ]
        
        for pattern, relation_type, confidence in relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Find entities that overlap with matched groups
                groups = match.groups()
                if len(groups) >= 2:
                    head_text = groups[0].strip()
                    tail_text = groups[1].strip()
                    
                    # Find corresponding entities
                    head_entity = None
                    tail_entity = None
                    
                    for start, end, cid, label, ent_text in entity_positions:
                        if head_text.lower() in ent_text.lower() or ent_text.lower() in head_text.lower():
                            head_entity = cid
                        if tail_text.lower() in ent_text.lower() or ent_text.lower() in tail_text.lower():
                            tail_entity = cid
                    
                    if head_entity is not None and tail_entity is not None and head_entity != tail_entity:
                        relation = RuleRelation(
                            rid=relation_id,
                            head_cid=head_entity,
                            tail_cid=tail_entity,
                            label=relation_type,
                            evidence=match.group(),
                            confidence=confidence,
                            rule_pattern=pattern
                        )
                        
                        relations.append(relation)
                        relation_id += 1
                        
                        # Update statistics
                        self.stats["relations_matched"] += 1
                        self.stats["rules_fired"][f"relation_{relation_type}"] += 1
        
        return relations
    
    def process_sentence(self, 
                        doc_id: str,
                        sent_id: str, 
                        text: str,
                        sent_start_offset: int = 0) -> Dict[str, Any]:
        """
        Process a sentence with rule-based annotation.
        
        Args:
            doc_id: Document ID
            sent_id: Sentence ID  
            text: Sentence text
            sent_start_offset: Character offset in document
            
        Returns:
            Rule-based annotation result
        """
        # Extract entities
        entities = self.extract_entities(text, sent_start_offset)
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        
        # Build result
        result = {
            "doc_id": doc_id,
            "sent_id": sent_id,
            "rule_results": {
                "entities": [asdict(e) for e in entities],
                "relations": [asdict(r) for r in relations]
            },
            "model_info": {
                "model": "rule_based_v1",
                "version": "1.0",
                "rules_fired": dict(self.stats["rules_fired"])
            },
            "processing_stats": {
                "entities_found": len(entities),
                "relations_found": len(relations)
            }
        }
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return dict(self.stats)
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            "entities_matched": 0,
            "relations_matched": 0,
            "rules_fired": defaultdict(int)
        }


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rule-based annotation engine")
    parser.add_argument("--text", help="Test text")
    parser.add_argument("--input-file", help="Input file to process")
    parser.add_argument("--output-file", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = ShimpAquacultureRuleEngine()
    
    if args.text:
        # Test single text
        result = engine.process_sentence("test_doc", "test_sent", args.text)
        print(json.dumps(result, indent=2))
        
    elif args.input_file:
        # Process file
        results = []
        with open(args.input_file, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    result = engine.process_sentence(f"doc_{i}", f"s_{i}", line.strip())
                    results.append(result)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Processed {len(results)} sentences to {args.output_file}")
        else:
            for result in results:
                print(json.dumps(result))
        
        # Print statistics
        stats = engine.get_statistics()
        print(f"\nRule Engine Statistics:")
        print(f"  Entities matched: {stats['entities_matched']}")
        print(f"  Relations matched: {stats['relations_matched']}")
        print(f"  Rules fired: {len(stats['rules_fired'])}")
    
    else:
        # Interactive mode
        print("Rule-based annotation engine ready. Enter text (Ctrl+C to exit):")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    result = engine.process_sentence("interactive", "s1", text)
                    entities = result["rule_results"]["entities"]
                    relations = result["rule_results"]["relations"]
                    
                    print(f"Found {len(entities)} entities, {len(relations)} relations")
                    for ent in entities:
                        print(f"  {ent['label']}: {ent['text']} (conf: {ent['confidence']:.2f})")
                    for rel in relations:
                        print(f"  Relation: {rel['label']} (conf: {rel['confidence']:.2f})")
        except KeyboardInterrupt:
            print("\nGoodbye!")
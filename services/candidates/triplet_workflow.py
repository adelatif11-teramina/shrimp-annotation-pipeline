"""Triplet workflow orchestration for the shrimp annotation pipeline.

Coordinates multi-agent processing: primary GPT extraction, auditing agent,
rule-engine cross checks, and packaging for human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple

from services.candidates.llm_candidate_generator import (
    LLMCandidateGenerator,
    EntityCandidate,
    TripletCandidate,
    TripletAuditEntry,
    TripletAuditReport,
    TripletExtractionResult,
)
from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine


@dataclass
class TripletReviewItem:
    """Triplet package ready for UI review."""

    triplet_id: str
    relation: str
    head: Dict[str, Any]
    tail: Dict[str, Any]
    evidence: str
    confidence: float
    audit: Optional[TripletAuditEntry] = None
    rule_support: bool = False
    rule_sources: List[str] = field(default_factory=list)
    raw_triplet: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "triplet_id": self.triplet_id,
            "relation": self.relation,
            "head": self.head,
            "tail": self.tail,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "rule_support": self.rule_support,
            "rule_sources": list(self.rule_sources),
            "raw_triplet": self.raw_triplet,
        }
        if self.audit:
            payload["audit"] = self.audit.to_dict()
        return payload


@dataclass
class TripletWorkflowResult:
    """Final artifact for a sentence combining agents and cross-checks."""

    doc_id: str
    sent_id: str
    sentence: str
    triplets: List[TripletReviewItem]
    audit_overall_verdict: str
    audit_notes: Optional[str]
    entities: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    rule_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "sent_id": self.sent_id,
            "sentence": self.sentence,
            "triplets": [item.to_dict() for item in self.triplets],
            "audit_overall_verdict": self.audit_overall_verdict,
            "audit_notes": self.audit_notes,
            "entities": self.entities,
            "nodes": self.nodes,
            "links": self.links,
            "rule_result": self.rule_result or {},
        }


class TripletWorkflow:
    """Helper that ties together GPT extraction, auditing, and rule checks."""

    def __init__(
        self,
        llm_generator: LLMCandidateGenerator,
        rule_engine: Optional[ShimpAquacultureRuleEngine] = None,
    ) -> None:
        self.llm_generator = llm_generator
        self.rule_engine = rule_engine or ShimpAquacultureRuleEngine()

    @staticmethod
    def _match_entity(node_label: str, node_type: str, entities: List[EntityCandidate]) -> Optional[EntityCandidate]:
        label_lower = (node_label or "").lower()
        for entity in entities:
            if entity.label == node_type and entity.text.lower() == label_lower:
                return entity
        # Fallback: partial match when text differs slightly
        for entity in entities:
            if entity.label == node_type and label_lower in entity.text.lower():
                return entity
        return None

    @staticmethod
    def _entity_payload(node_id: str, node_conf: float, entity: Optional[EntityCandidate], node_label: str, node_type: str, start: Optional[int], end: Optional[int]) -> Dict[str, Any]:
        if entity:
            payload = asdict(entity)
        else:
            payload = {
                "cid": None,
                "text": node_label,
                "label": node_type,
                "start": start,
                "end": end,
                "confidence": node_conf,
            }
        payload.update({
            "node_id": node_id,
            "node_confidence": node_conf,
        })
        return payload

    @staticmethod
    def _build_rule_maps(rule_result: Optional[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
        if not rule_result:
            return {}, []
        entity_map = {entity["cid"]: entity for entity in rule_result.get("entities", []) if "cid" in entity}
        relations = rule_result.get("relations", [])
        return entity_map, relations

    @staticmethod
    def _check_rule_support(
        triplet: TripletCandidate,
        head_payload: Dict[str, Any],
        tail_payload: Dict[str, Any],
        rule_entities: Dict[int, Dict[str, Any]],
        rule_relations: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        if not rule_relations:
            return False, []

        matched_rules: List[str] = []
        relation_lower = triplet.relation.lower()
        head_text = head_payload.get("text", "").lower()
        tail_text = tail_payload.get("text", "").lower()

        for relation in rule_relations:
            head_entity = rule_entities.get(relation.get("head_cid"))
            tail_entity = rule_entities.get(relation.get("tail_cid"))
            if not head_entity or not tail_entity:
                continue
            if relation.get("label", "").lower() != relation_lower:
                continue
            if head_entity.get("text", "").lower() != head_text:
                continue
            if tail_entity.get("text", "").lower() != tail_text:
                continue
            matched_rules.append(relation.get("rule_pattern", relation.get("label")))

        return bool(matched_rules), matched_rules

    async def process_sentence(
        self,
        doc_id: str,
        sent_id: str,
        sentence: str,
        title: Optional[str] = None,
    ) -> TripletWorkflowResult:
        """Run the full triplet workflow for a sentence."""
        entities = await self.llm_generator.extract_entities(sentence)
        triplet_extraction: TripletExtractionResult = await self.llm_generator.generate_triplets(sentence)
        audit_report: TripletAuditReport = await self.llm_generator.audit_triplets(sentence, triplet_extraction.triplets)

        rule_result = None
        if self.rule_engine:
            rule_result = self.rule_engine.process_sentence(doc_id, sent_id, sentence)

        rule_entity_map, rule_relations = self._build_rule_maps(rule_result)
        audit_lookup = {entry.triplet_id: entry for entry in audit_report.triplets}

        review_items: List[TripletReviewItem] = []
        for triplet in triplet_extraction.triplets:
            head_match = self._match_entity(triplet.head.label, triplet.head.type, entities)
            tail_match = self._match_entity(triplet.tail.label, triplet.tail.type, entities)

            head_payload = self._entity_payload(
                triplet.head.node_id,
                triplet.head.confidence,
                head_match,
                triplet.head.label,
                triplet.head.type,
                triplet.head.start,
                triplet.head.end,
            )
            tail_payload = self._entity_payload(
                triplet.tail.node_id,
                triplet.tail.confidence,
                tail_match,
                triplet.tail.label,
                triplet.tail.type,
                triplet.tail.start,
                triplet.tail.end,
            )

            rule_support, rule_sources = self._check_rule_support(
                triplet,
                head_payload,
                tail_payload,
                rule_entity_map,
                rule_relations,
            )

            review_items.append(
                TripletReviewItem(
                    triplet_id=triplet.triplet_id,
                    relation=triplet.relation,
                    head=head_payload,
                    tail=tail_payload,
                    evidence=triplet.evidence,
                    confidence=triplet.confidence,
                    audit=audit_lookup.get(triplet.triplet_id),
                    rule_support=rule_support,
                    rule_sources=rule_sources,
                    raw_triplet=triplet.to_dict(),
                )
            )

        workflow_result = TripletWorkflowResult(
            doc_id=doc_id,
            sent_id=sent_id,
            sentence=sentence,
            triplets=review_items,
            audit_overall_verdict=audit_report.overall_verdict,
            audit_notes=audit_report.notes,
            entities=[asdict(entity) for entity in entities],
            nodes=[asdict(node) for node in triplet_extraction.nodes],
            links=[{
                "link_id": link.link_id,
                "source": link.source,
                "target": link.target,
                "relation": link.relation,
                "evidence": link.evidence,
                "confidence": link.confidence,
            } for link in triplet_extraction.links],
            rule_result=rule_result,
        )
        return workflow_result

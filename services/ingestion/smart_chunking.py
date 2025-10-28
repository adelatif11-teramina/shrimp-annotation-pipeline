"""
Smart Short-Paragraph Chunking for Improved Annotation Quality

This module implements an intelligent chunking strategy that addresses the context loss
problem in sentence-level annotation while avoiding the cognitive overload of full paragraphs.
"""

import re
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

# Minimal sentence class to avoid circular imports
@dataclass
class Sentence:
    sent_id: str
    start: int
    end: int
    text: str


@dataclass 
class SmartChunk:
    """A semantically coherent chunk optimized for annotation"""
    chunk_id: str
    start: int
    end: int
    text: str
    sentences: List[Sentence]
    chunk_type: str = "smart_paragraph"
    
    # Context preservation
    has_definitions: bool = False
    has_pronouns: bool = False  
    has_cross_references: bool = False
    discourse_markers: List[str] = field(default_factory=list)
    
    # Quality metrics
    sentence_count: int = 0
    char_count: int = 0
    estimated_entities: int = 0
    
    def __post_init__(self):
        self.sentence_count = len(self.sentences)
        self.char_count = len(self.text)
        self._analyze_context_features()
    
    def _analyze_context_features(self):
        """Analyze chunk for context preservation features"""
        text_lower = self.text.lower()
        
        # Check for definitions (parentheses, "called", "known as")
        self.has_definitions = bool(re.search(r'\([^)]{2,}\)|called|known as|defined as', self.text))
        
        # Check for pronouns that need context
        pronouns = ['this', 'these', 'that', 'those', 'it', 'they', 'here', 'there']
        self.has_pronouns = any(re.search(rf'\b{pronoun}\b', text_lower) for pronoun in pronouns)
        
        # Check for cross-references
        cross_refs = ['above', 'below', 'previous', 'following', 'aforementioned', 'latter', 'former']
        self.has_cross_references = any(ref in text_lower for ref in cross_refs)
        
        # Identify discourse markers
        markers = ['however', 'therefore', 'meanwhile', 'moreover', 'furthermore', 'in addition', 'consequently']
        self.discourse_markers = [m for m in markers if m in text_lower]


class SmartChunkingService:
    """Service for intelligent short-paragraph chunking"""
    
    def __init__(self, target_length: Tuple[int, int] = (150, 400)):
        """
        Initialize smart chunking service.
        
        Args:
            target_length: (min_chars, max_chars) for optimal chunk size
        """
        self.min_length, self.max_length = target_length
    
    def segment_sentences(self, text: str) -> List[Sentence]:
        """Simple sentence segmentation for smart chunking"""
        if not text:
            return []
        
        # Simple regex-based sentence splitting
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sent_texts = re.split(pattern, text)
        
        sentences = []
        offset = 0
        for i, sent_text in enumerate(sent_texts):
            if not sent_text.strip():
                continue
                
            start = text.find(sent_text, offset)
            if start == -1:
                start = offset
            end = start + len(sent_text)
            
            sentences.append(Sentence(
                sent_id=f"s{i}",
                start=start,
                end=end,
                text=sent_text.strip()
            ))
            offset = end
        
        return sentences
    
    def create_smart_chunks(self, text: str) -> List[SmartChunk]:
        """
        Create smart chunks that preserve context and optimize annotation quality.
        
        Strategy:
        1. Start with sentence segmentation
        2. Group sentences semantically
        3. Ensure optimal length (150-400 chars)
        4. Preserve definitions and references
        5. Handle cross-sentence relations
        """
        # Get sentence segmentation first
        sentences = self.segment_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_group = []
        chunk_id = 0
        
        i = 0
        while i < len(sentences):
            current_group = [sentences[i]]
            current_length = len(sentences[i].text)
            
            # Strategy 1: Definition Preservation
            # If current sentence has definition, try to include it complete
            if self._has_definition_pattern(sentences[i].text):
                chunk = self._create_definition_chunk(sentences, i, chunk_id)
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
                    i += len(chunk.sentences)
                    continue
            
            # Strategy 2: Smart Grouping with Lookahead
            # Look ahead to group related sentences
            j = i + 1
            while j < len(sentences) and current_length < self.max_length:
                next_sentence = sentences[j]
                next_length = current_length + len(next_sentence.text) + 1  # +1 for space
                
                # Stop if adding next sentence exceeds max length
                if next_length > self.max_length:
                    break
                
                # Check if next sentence should be grouped
                should_group = self._should_group_sentences(
                    current_group[-1], next_sentence, current_group
                )
                
                if should_group:
                    current_group.append(next_sentence)
                    current_length = next_length
                    j += 1
                else:
                    break
            
            # Strategy 3: Minimum Length Enforcement  
            # Ensure chunks meet minimum length unless at document end
            if current_length < self.min_length and j < len(sentences):
                # Try to add one more sentence if possible
                if j < len(sentences):
                    next_length = current_length + len(sentences[j].text) + 1
                    if next_length <= self.max_length:
                        current_group.append(sentences[j])
                        j += 1
            
            # Create chunk from current group
            chunk = self._create_chunk_from_sentences(current_group, chunk_id)
            chunks.append(chunk)
            chunk_id += 1
            i = j
        
        return self._post_process_chunks(chunks)
    
    def _has_definition_pattern(self, text: str) -> bool:
        """Check if sentence contains definition patterns"""
        patterns = [
            r'\([^)]{3,}\)',  # Parenthetical definitions
            r'\b(?:called|known as|defined as|referred to as)\b',
            r'\b[A-Z]{2,}\b.*\(',  # Acronyms with definitions
            r'[:,]\s*[a-z].*[,;]'  # Colon or comma explanations
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _create_definition_chunk(self, sentences: List[Sentence], start_idx: int, chunk_id: int) -> Optional[SmartChunk]:
        """Create a chunk that preserves definition context"""
        current_group = [sentences[start_idx]]
        
        # Look for related context in next sentence
        if start_idx + 1 < len(sentences):
            next_sent = sentences[start_idx + 1] 
            combined_length = len(current_group[0].text) + len(next_sent.text) + 1
            
            # Include next sentence if it provides context and fits
            if (combined_length <= self.max_length and 
                self._provides_definition_context(current_group[0].text, next_sent.text)):
                current_group.append(next_sent)
        
        return self._create_chunk_from_sentences(current_group, chunk_id)
    
    def _provides_definition_context(self, first_text: str, second_text: str) -> bool:
        """Check if second sentence provides context for first"""
        # Check for pronouns referring back
        second_lower = second_text.lower()
        context_indicators = [
            'this', 'these', 'that', 'those', 'it', 'they',
            'the disease', 'the pathogen', 'the condition', 'the syndrome'
        ]
        return any(indicator in second_lower for indicator in context_indicators)
    
    def _should_group_sentences(self, prev_sent: Sentence, next_sent: Sentence, current_group: List[Sentence]) -> bool:
        """Determine if sentences should be grouped together"""
        prev_text = prev_sent.text.lower()
        next_text = next_sent.text.lower()
        
        # Group if next sentence starts with discourse connector
        connectors = ['however', 'therefore', 'meanwhile', 'moreover', 'furthermore', 'in addition']
        if any(next_text.startswith(conn) for conn in connectors):
            return True
        
        # Group if next sentence has pronouns referring to previous
        pronouns = ['this', 'that', 'it', 'they', 'these', 'those']
        if any(next_text.startswith(pron) for pron in pronouns):
            return True
        
        # Group if continuing same topic (simple heuristic)
        # Check for shared domain terms
        domain_terms = ['shrimp', 'vibrio', 'pathogen', 'disease', 'larvae', 'mortality', 'infection']
        prev_terms = [term for term in domain_terms if term in prev_text]
        next_terms = [term for term in domain_terms if term in next_text]
        shared_terms = set(prev_terms) & set(next_terms)
        
        if len(shared_terms) >= 1:
            return True
        
        # Don't group if next sentence starts new topic
        topic_starters = ['in conclusion', 'to summarize', 'next', 'first', 'second', 'finally']
        if any(next_text.startswith(starter) for starter in topic_starters):
            return False
        
        return False
    
    def _create_chunk_from_sentences(self, sentences: List[Sentence], chunk_id: int) -> SmartChunk:
        """Create a SmartChunk from a list of sentences"""
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")
        
        # Calculate span
        start = sentences[0].start
        end = sentences[-1].end
        
        # Reconstruct text (preserve original spacing)
        chunk_text = ""
        for i, sent in enumerate(sentences):
            if i == 0:
                chunk_text = sent.text
            else:
                # Preserve original spacing between sentences
                prev_end = sentences[i-1].end
                curr_start = sent.start
                spacing = " " if curr_start - prev_end <= 2 else "  "
                chunk_text += spacing + sent.text
        
        return SmartChunk(
            chunk_id=f"sc{chunk_id}",
            start=start,
            end=end,
            text=chunk_text,
            sentences=sentences
        )
    
    def _post_process_chunks(self, chunks: List[SmartChunk]) -> List[SmartChunk]:
        """Post-process chunks for quality optimization"""
        # Merge very short adjacent chunks if beneficial
        optimized = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If current chunk is very short and can be merged with next
            if (i + 1 < len(chunks) and 
                current.char_count < self.min_length and
                current.char_count + chunks[i + 1].char_count <= self.max_length):
                
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_sentences = current.sentences + next_chunk.sentences
                merged_text = current.text + " " + next_chunk.text
                
                merged = SmartChunk(
                    chunk_id=f"sc{len(optimized)}",
                    start=current.start,
                    end=next_chunk.end,
                    text=merged_text,
                    sentences=merged_sentences
                )
                optimized.append(merged)
                i += 2  # Skip next chunk since it's merged
            else:
                optimized.append(current)
                i += 1
        
        return optimized
    
    def get_chunking_statistics(self, chunks: List[SmartChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking quality"""
        if not chunks:
            return {}
        
        char_counts = [chunk.char_count for chunk in chunks]
        sentence_counts = [chunk.sentence_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chars_per_chunk": sum(char_counts) / len(char_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "avg_sentences_per_chunk": sum(sentence_counts) / len(sentence_counts),
            "chunks_with_definitions": sum(1 for c in chunks if c.has_definitions),
            "chunks_with_pronouns": sum(1 for c in chunks if c.has_pronouns),
            "chunks_with_cross_refs": sum(1 for c in chunks if c.has_cross_references),
            "optimal_size_chunks": sum(1 for c in chunks if self.min_length <= c.char_count <= self.max_length),
            "quality_score": sum(1 for c in chunks if self.min_length <= c.char_count <= self.max_length) / len(chunks)
        }


# Integration with existing Document class
def integrate_smart_chunking():
    """Helper to show how to integrate with existing Document class"""
    from services.ingestion.document_ingestion import Document
    
    # Add smart chunking to Document class
    def add_smart_chunks(self, target_length: Tuple[int, int] = (150, 400)):
        """Add smart chunks to document"""
        chunker = SmartChunkingService(target_length)
        self.smart_chunks = chunker.create_smart_chunks(self.raw_text)
        self.chunks = self.smart_chunks  # Set as primary chunks
        self.chunk_type = "smart_paragraph"
        return self.smart_chunks
    
    # This would be added to Document class in practice
    Document.add_smart_chunks = add_smart_chunks
"""
Integration of Smart Short-Paragraph Chunking into Existing Pipeline

This module shows how to integrate the new smart chunking approach into the 
existing annotation pipeline with minimal code changes.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

# Handle imports safely for Railway deployment
try:
    from services.ingestion.document_ingestion import DocumentIngestionService, Document
    HAS_DOCUMENT_INGESTION = True
except ImportError:
    HAS_DOCUMENT_INGESTION = False
    print("Warning: Document ingestion service not available")

try:
    from services.ingestion.smart_chunking import SmartChunkingService, SmartChunk
    HAS_SMART_CHUNKING = True
except ImportError:
    HAS_SMART_CHUNKING = False
    print("Warning: Smart chunking service not available")


# Only define the class if dependencies are available
if HAS_DOCUMENT_INGESTION and HAS_SMART_CHUNKING:
    class ImprovedDocumentIngestionService(DocumentIngestionService):
        """Enhanced document ingestion with smart chunking support"""
        
        def __init__(self, 
                 data_training_path: Optional[Path] = None,
                 segmenter: str = "regex",
                 chunking_mode: str = "smart_paragraph",  # New default!
                 enable_llm_preprocessing: bool = False,
                 openai_api_key: Optional[str] = None,
                 smart_chunk_length: tuple = (150, 400)):
        """
        Initialize enhanced ingestion service with smart chunking.
        
        Args:
            chunking_mode: "sentence", "paragraph", or "smart_paragraph" (recommended)
            smart_chunk_length: (min_chars, max_chars) for smart chunks
        """
        super().__init__(data_training_path, segmenter, chunking_mode, 
                        enable_llm_preprocessing, openai_api_key)
        
        # Add smart chunking capability
        if chunking_mode == "smart_paragraph":
            self.smart_chunker = SmartChunkingService(target_length=smart_chunk_length)
        else:
            self.smart_chunker = None
    
    def ingest_text_file(self, 
                        file_path: Path,
                        source: str = "paper",
                        title: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Document:
        """Enhanced ingestion with smart chunking support"""
        
        # Use parent method for basic processing
        doc = super().ingest_text_file(file_path, source, title, metadata)
        
        # Apply smart chunking if requested
        if self.chunking_mode == "smart_paragraph" and self.smart_chunker:
            smart_chunks = self.smart_chunker.create_smart_chunks(doc.raw_text)
            
            # Replace standard chunks with smart chunks
            doc.smart_chunks = smart_chunks
            doc.chunks = smart_chunks
            doc.chunk_type = "smart_paragraph"
            doc.chunking_mode = "smart_paragraph"
            
            # Update metadata with smart chunking stats
            stats = self.smart_chunker.get_chunking_statistics(smart_chunks)
            doc.metadata.update({
                'smart_chunking_stats': stats,
                'chunk_count': len(smart_chunks),
                'avg_chunk_length': stats.get('avg_chars_per_chunk', 0),
                'context_preservation': {
                    'chunks_with_definitions': stats.get('chunks_with_definitions', 0),
                    'chunks_with_pronouns': stats.get('chunks_with_pronouns', 0),
                    'quality_score': stats.get('quality_score', 0)
                }
            })
        
        return doc


def migrate_to_smart_chunking():
    """
    Migration guide: How to switch existing pipeline to smart chunking
    """
    migration_steps = """
    🔄 MIGRATION TO SMART CHUNKING:
    
    1. Update ingestion service initialization:
       OLD: service = DocumentIngestionService(chunking_mode='sentence')  
       NEW: service = ImprovedDocumentIngestionService(chunking_mode='smart_paragraph')
    
    2. Update annotation API initialization:
       In services/api/annotation_api.py, line ~282:
       OLD: ingestion_service = DocumentIngestionService()
       NEW: ingestion_service = ImprovedDocumentIngestionService(chunking_mode='smart_paragraph')
    
    3. Update configuration files:
       In config/local_config.yaml, add:
       chunking:
         mode: smart_paragraph
         target_length: [150, 400]
         preserve_context: true
    
    4. Update frontend to handle smart chunks:
       In ui/src/components/, update chunk display components to use:
       - chunk.chunk_id instead of sent_id
       - chunk.text for full chunk text
       - chunk.sentences for individual sentences within chunk
    
    5. Update database schema (if storing chunks):
       Add columns: chunk_type, context_features, quality_metrics
    
    6. Batch migration of existing documents:
       Run migration script to re-chunk existing documents with smart chunking
    """
    print(migration_steps)


def demo_annotation_quality_improvement():
    """
    Demonstrate the annotation quality improvement with smart chunking
    """
    
    # Sample problematic text from real scientific papers
    problem_text = '''
    Vibrio parahaemolyticus causes acute hepatopancreatic necrosis disease (AHPND) in shrimp. 
    This pathogen has emerged as a major threat to aquaculture. 
    The disease results in significant economic losses worldwide.
    
    Recent studies have identified novel virulence factors. 
    These factors contribute to the pathogen's increased virulence. 
    Understanding their mechanisms is crucial for disease control.
    '''
    
    print("=== ANNOTATION QUALITY COMPARISON ===\n")
    
    # 1. Current sentence-level approach
    sentence_service = DocumentIngestionService(chunking_mode='sentence')
    sentences = sentence_service.segment_sentences(problem_text.strip())
    
    print("❌ SENTENCE-LEVEL PROBLEMS:")
    print(f"  s0: '{sentences[0].text}'")
    print(f"  s1: '{sentences[1].text}' ← 'This pathogen' needs context!")
    print(f"  s2: '{sentences[2].text}' ← 'The disease' ambiguous!")
    print(f"  s4: '{sentences[4].text}' ← 'These factors' unclear!")
    print()
    
    # 2. Smart chunking solution
    smart_service = ImprovedDocumentIngestionService(chunking_mode='smart_paragraph')
    doc = smart_service.ingest_text_file(
        Path('/dev/null'),  # Dummy path
        source='demo'
    )
    
    # Override with our test text
    doc.raw_text = problem_text.strip()
    smart_chunks = smart_service.smart_chunker.create_smart_chunks(doc.raw_text)
    
    print("✅ SMART CHUNKING SOLUTION:")
    for i, chunk in enumerate(smart_chunks):
        print(f"  sc{i}: '{chunk.text}'")
        context_features = []
        if chunk.has_definitions:
            context_features.append("preserves definitions")
        if chunk.has_pronouns:
            context_features.append("resolves pronouns")
        if context_features:
            print(f"        → {', '.join(context_features)}")
    print()
    
    # Calculate improvement metrics
    stats = smart_service.smart_chunker.get_chunking_statistics(smart_chunks)
    
    print("📊 QUALITY IMPROVEMENT:")
    print(f"  • Context preservation: {stats['chunks_with_pronouns']}/{stats['total_chunks']} chunks preserve pronoun context")
    print(f"  • Optimal size: {stats['optimal_size_chunks']}/{stats['total_chunks']} chunks in target range")
    print(f"  • Quality score: {stats['quality_score']:.1%}")
    print(f"  • Reduced cognitive load: {stats['avg_sentences_per_chunk']:.1f} sentences per chunk vs 1.0 for sentence-level")
    print()
    
    print("🎯 ANNOTATION BENEFITS:")
    print("  • Annotators can understand entity references in context")
    print("  • Cross-sentence relations are preserved")
    print("  • Definitions stay with their usage")
    print("  • Reduced ambiguity from pronouns and references")
    print("  • Manageable chunk size prevents cognitive overload")


if __name__ == "__main__":
    print("Smart Chunking Integration Demo\n")
    demo_annotation_quality_improvement()
    print("\n" + "="*60 + "\n")
    migrate_to_smart_chunking()
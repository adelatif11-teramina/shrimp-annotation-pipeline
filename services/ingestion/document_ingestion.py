"""
Document Ingestion Service

Ingests documents from various sources including the data-training pipeline outputs.
Performs sentence segmentation and prepares documents for annotation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# NLP libraries for sentence segmentation
try:
    import spacy
    HAS_SPACY = True
except (ImportError, ValueError) as e:
    # ValueError occurs with NumPy 2.x incompatibility
    if "numpy.dtype size changed" in str(e):
        logger.warning("spaCy incompatible with NumPy 2.x, using fallback segmentation")
    HAS_SPACY = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

@dataclass
class Sentence:
    """Represents a sentence in a document"""
    sent_id: str
    start: int
    end: int
    text: str
    paragraph_id: Optional[int] = None
    metadata: Optional[Dict] = None

@dataclass
class Document:
    """Represents an ingested document"""
    doc_id: str
    source: str  # paper, report, hatchery_log, manual, dataset
    title: Optional[str]
    pub_date: Optional[str]
    raw_text: str
    sentences: List[Sentence]
    metadata: Dict[str, Any]

class DocumentIngestionService:
    """
    Service for ingesting and processing documents.
    
    Features:
    - Multiple input formats (txt, json, jsonl)
    - Integration with data-training pipeline outputs
    - Sentence segmentation with offset preservation
    - Document deduplication
    - Metadata extraction
    """
    
    def __init__(self, 
                 data_training_path: Optional[Path] = None,
                 segmenter: str = "regex"):
        """
        Initialize the ingestion service.
        
        Args:
            data_training_path: Path to data-training project
            segmenter: Sentence segmentation method (regex, spacy, nltk)
        """
        # Setup path to sibling project
        if data_training_path:
            self.data_training_path = Path(data_training_path)
        else:
            # Default to sibling project
            self.data_training_path = Path(__file__).parent.parent.parent.parent / "data-training"
        
        # Setup sentence segmenter
        self.segmenter = segmenter
        if segmenter == "spacy" and HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, falling back to regex")
                self.segmenter = "regex"
        elif segmenter == "nltk" and HAS_NLTK:
            try:
                nltk.download('punkt', quiet=True)
            except:
                logger.warning("NLTK data not available, falling back to regex")
                self.segmenter = "regex"
        
        logger.info(f"Initialized ingestion service with {self.segmenter} segmenter")
    
    def generate_doc_id(self, text: str, title: Optional[str] = None) -> str:
        """Generate unique document ID"""
        content = f"{title or ''}:{text[:1000]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def segment_sentences(self, text: str) -> List[Sentence]:
        """
        Segment text into sentences with offset preservation.
        
        Args:
            text: Input text
            
        Returns:
            List of Sentence objects with offsets
        """
        sentences = []
        
        if self.segmenter == "spacy" and HAS_SPACY:
            doc = self.nlp(text)
            for i, sent in enumerate(doc.sents):
                sentences.append(Sentence(
                    sent_id=f"s{i}",
                    start=sent.start_char,
                    end=sent.end_char,
                    text=sent.text.strip()
                ))
        
        elif self.segmenter == "nltk" and HAS_NLTK:
            sent_texts = sent_tokenize(text)
            offset = 0
            for i, sent_text in enumerate(sent_texts):
                start = text.find(sent_text, offset)
                end = start + len(sent_text)
                sentences.append(Sentence(
                    sent_id=f"s{i}",
                    start=start,
                    end=end,
                    text=sent_text.strip()
                ))
                offset = end
        
        else:
            # Regex-based fallback
            # Pattern for sentence boundaries
            pattern = r'(?<=[.!?])\s+(?=[A-Z])'
            sent_texts = re.split(pattern, text)
            
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
        
        # Add paragraph information
        self._add_paragraph_info(sentences, text)
        
        return sentences
    
    def _add_paragraph_info(self, sentences: List[Sentence], text: str):
        """Add paragraph IDs to sentences"""
        paragraphs = text.split('\n\n')
        para_boundaries = []
        offset = 0
        
        for para in paragraphs:
            if para.strip():
                start = text.find(para, offset)
                end = start + len(para)
                para_boundaries.append((start, end))
                offset = end
        
        # Assign paragraph IDs to sentences
        for sent in sentences:
            for i, (start, end) in enumerate(para_boundaries):
                if sent.start >= start and sent.end <= end:
                    sent.paragraph_id = i
                    break
    
    def ingest_text_file(self, 
                        file_path: Path,
                        source: str = "paper",
                        title: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Document:
        """
        Ingest a plain text file.
        
        Args:
            file_path: Path to text file
            source: Document source type
            title: Document title
            metadata: Additional metadata
            
        Returns:
            Document object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Generate document ID
        doc_id = self.generate_doc_id(text, title or file_path.stem)
        
        # Segment sentences
        sentences = self.segment_sentences(text)
        
        # Build document
        doc = Document(
            doc_id=doc_id,
            source=source,
            title=title or file_path.stem,
            pub_date=None,
            raw_text=text,
            sentences=sentences,
            metadata=metadata or {}
        )
        
        doc.metadata['file_path'] = str(file_path)
        doc.metadata['ingestion_time'] = datetime.now().isoformat()
        doc.metadata['sentence_count'] = len(sentences)
        
        return doc
    
    def ingest_from_data_training(self, 
                                  file_name: str,
                                  use_processed: bool = True) -> Optional[Document]:
        """
        Ingest document from data-training pipeline.
        
        Args:
            file_name: Name of file (without extension)
            use_processed: Use processed text vs raw PDF text
            
        Returns:
            Document object or None if not found
        """
        if use_processed:
            # Use OCR-processed text
            text_path = self.data_training_path / "data/output/text" / f"{file_name}.txt"
        else:
            # Would need PDF processing
            text_path = None
            logger.warning("Raw PDF ingestion not implemented")
            return None
        
        if text_path and text_path.exists():
            # Check for existing NER/RE annotations
            metadata = {}
            annotation_path = self.data_training_path / "data/data_training" / f"{file_name}_ner_relations.json"
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    metadata['existing_annotations'] = json.load(f)
            
            return self.ingest_text_file(
                text_path,
                source="paper",
                title=file_name,
                metadata=metadata
            )
        else:
            logger.error(f"File not found: {text_path}")
            return None
    
    def ingest_jsonl_annotations(self, file_path: Path) -> List[Document]:
        """
        Ingest existing JSONL annotations from data-training.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of Document objects
        """
        documents = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    
                    # Extract text and create minimal document
                    text = data.get("text", "")
                    doc_id = f"{file_path.stem}_{line_num}"
                    
                    # Create single sentence (already annotated segment)
                    sentences = [Sentence(
                        sent_id="s0",
                        start=0,
                        end=len(text),
                        text=text
                    )]
                    
                    doc = Document(
                        doc_id=doc_id,
                        source="dataset",
                        title=None,
                        pub_date=None,
                        raw_text=text,
                        sentences=sentences,
                        metadata={
                            "line_number": line_num,
                            "existing_entities": data.get("entities", []),
                            "existing_relations": data.get("relations", [])
                        }
                    )
                    
                    documents.append(doc)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num}: {e}")
        
        return documents
    
    def batch_ingest_from_directory(self, 
                                   directory: Path,
                                   pattern: str = "*.txt") -> List[Document]:
        """
        Ingest multiple documents from a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in directory.glob(pattern):
            try:
                doc = self.ingest_text_file(file_path)
                documents.append(doc)
                logger.info(f"Ingested {file_path.name}: {len(doc.sentences)} sentences")
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
        
        return documents
    
    def export_for_annotation(self, 
                             document: Document,
                             output_path: Path,
                             format: str = "jsonl"):
        """
        Export document for annotation.
        
        Args:
            document: Document to export
            output_path: Output file path
            format: Export format (jsonl, json, txt)
        """
        if format == "jsonl":
            # Export as JSONL (one sentence per line)
            with open(output_path, 'w') as f:
                for sent in document.sentences:
                    record = {
                        "doc_id": document.doc_id,
                        "sent_id": sent.sent_id,
                        "text": sent.text,
                        "title": document.title,
                        "metadata": {
                            "start": sent.start,
                            "end": sent.end,
                            "paragraph_id": sent.paragraph_id,
                            "source": document.source
                        }
                    }
                    f.write(json.dumps(record) + "\n")
        
        elif format == "json":
            # Export as single JSON
            with open(output_path, 'w') as f:
                json.dump(asdict(document), f, indent=2)
        
        elif format == "txt":
            # Export as plain text with sentence markers
            with open(output_path, 'w') as f:
                f.write(f"# Document: {document.doc_id}\n")
                f.write(f"# Title: {document.title}\n")
                f.write(f"# Sentences: {len(document.sentences)}\n\n")
                
                for sent in document.sentences:
                    f.write(f"[{sent.sent_id}] {sent.text}\n\n")
    
    def get_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about ingested documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Statistics dictionary
        """
        total_sentences = sum(len(doc.sentences) for doc in documents)
        sentence_lengths = [len(sent.text) for doc in documents for sent in doc.sentences]
        
        stats = {
            "document_count": len(documents),
            "total_sentences": total_sentences,
            "avg_sentences_per_doc": total_sentences / len(documents) if documents else 0,
            "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
            "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
            "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
            "sources": {}
        }
        
        # Count by source
        for doc in documents:
            stats["sources"][doc.source] = stats["sources"].get(doc.source, 0) + 1
        
        return stats


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document ingestion service")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "json", "txt"])
    parser.add_argument("--source", default="paper", 
                       choices=["paper", "report", "hatchery_log", "manual", "dataset"])
    parser.add_argument("--from-training", action="store_true",
                       help="Ingest from data-training project")
    
    args = parser.parse_args()
    
    # Initialize service
    service = DocumentIngestionService()
    
    # Process input
    input_path = Path(args.input)
    
    if args.from_training:
        # Ingest from data-training
        doc = service.ingest_from_data_training(input_path.stem)
        if doc:
            documents = [doc]
        else:
            documents = []
    elif input_path.is_file():
        # Single file
        doc = service.ingest_text_file(input_path, source=args.source)
        documents = [doc]
    else:
        # Directory
        documents = service.batch_ingest_from_directory(input_path)
    
    # Export if output specified
    if args.output and documents:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            output_path = output_dir / f"{doc.doc_id}.{args.format}"
            service.export_for_annotation(doc, output_path, args.format)
        
        print(f"Exported {len(documents)} documents to {output_dir}")
    
    # Print statistics
    stats = service.get_statistics(documents)
    print(f"\nIngestion Statistics:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Total sentences: {stats['total_sentences']}")
    print(f"  Avg sentences/doc: {stats['avg_sentences_per_doc']:.1f}")
    print(f"  Avg sentence length: {stats['avg_sentence_length']:.1f} chars")
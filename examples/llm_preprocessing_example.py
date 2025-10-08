#!/usr/bin/env python3
"""
Example: Using LLM Preprocessing for Document Ingestion

This example demonstrates how to use the LLM preprocessing feature to
automatically clean scientific documents before processing them through
the annotation pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingestion.document_ingestion import DocumentIngestionService

def example_with_preprocessing():
    """Example using LLM preprocessing to clean documents"""
    
    print("📄 LLM Preprocessing Example")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to use LLM preprocessing:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create sample document with bibliography
    sample_file = Path("sample_shrimp_paper.txt")
    sample_content = """
    Title: Vibrio Resistance in Pacific White Shrimp

    ABSTRACT
    We investigated genetic markers associated with Vibrio parahaemolyticus resistance in Penaeus vannamei populations. Our findings reveal key genomic regions linked to disease tolerance.

    INTRODUCTION  
    Penaeus vannamei (Pacific white shrimp) is economically important but susceptible to bacterial diseases. Vibrio parahaemolyticus causes significant mortality in aquaculture operations worldwide.

    METHODS
    DNA samples were collected from 500 individual shrimp across three farms in Thailand. SNP analysis was performed using Illumina sequencing technology.

    RESULTS
    We identified 12 SNP markers significantly associated with disease resistance. Resistant individuals showed 60% higher survival rates when challenged with Vibrio parahaemolyticus.

    DISCUSSION
    These genetic markers could be used for selective breeding programs to develop Vibrio-resistant shrimp lines.

    ACKNOWLEDGMENTS
    We thank the Thailand Department of Fisheries for sample collection permits. Dr. Johnson provided valuable statistical analysis support.

    REFERENCES
    1. Smith, A. et al. (2023) Genetic resistance in shrimp aquaculture. Marine Genomics 45:123-135.
    2. Jones, B. et al. (2022) Vibrio pathogenesis mechanisms. Aquaculture Research 78:456-467.
    3. Brown, C. (2021) SNP analysis in crustaceans. Molecular Biology Letters 12:89-101.

    FUNDING
    This work was supported by grants from the National Science Foundation (NSF-2023-0456) and the International Aquaculture Development Fund.

    AUTHOR INFORMATION
    Corresponding author: Dr. Example (example@university.edu)
    Department of Marine Biology, Example University, USA
    """
    
    with open(sample_file, 'w') as f:
        f.write(sample_content)
    
    try:
        print("1. Processing WITHOUT LLM preprocessing:")
        print("-" * 40)
        
        # Process without LLM preprocessing
        service_no_llm = DocumentIngestionService(
            enable_llm_preprocessing=False,
            chunking_mode="sentence"
        )
        
        doc_no_llm = service_no_llm.ingest_text_file(
            sample_file,
            title="Sample Shrimp Paper - No LLM"
        )
        
        print(f"   Text length: {len(doc_no_llm.raw_text)} characters")
        print(f"   Sentences: {len(doc_no_llm.sentences)}")
        print(f"   Contains 'REFERENCES': {'REFERENCES' in doc_no_llm.raw_text}")
        print(f"   Contains 'FUNDING': {'FUNDING' in doc_no_llm.raw_text}")
        
        print("\n2. Processing WITH LLM preprocessing:")
        print("-" * 40)
        
        # Process with LLM preprocessing
        service_with_llm = DocumentIngestionService(
            enable_llm_preprocessing=True,
            chunking_mode="sentence"
        )
        
        doc_with_llm = service_with_llm.ingest_text_file(
            sample_file,
            title="Sample Shrimp Paper - With LLM"
        )
        
        preprocessing_stats = doc_with_llm.metadata.get('preprocessing', {})
        
        print(f"   Original length: {preprocessing_stats.get('original_length', 0)} characters")
        print(f"   Cleaned length: {preprocessing_stats.get('cleaned_length', 0)} characters")
        print(f"   Reduction: {preprocessing_stats.get('reduction_ratio', 0):.1%}")
        print(f"   Sentences: {len(doc_with_llm.sentences)}")
        print(f"   Contains 'REFERENCES': {'REFERENCES' in doc_with_llm.raw_text}")
        print(f"   Contains 'FUNDING': {'FUNDING' in doc_with_llm.raw_text}")
        print(f"   Contains scientific content: {'Penaeus vannamei' in doc_with_llm.raw_text}")
        
        print("\n3. Sample cleaned sentences:")
        print("-" * 40)
        for i, sentence in enumerate(doc_with_llm.sentences[:5]):
            print(f"   {sentence.sent_id}: {sentence.text[:80]}...")
        
        print("\n✅ LLM preprocessing successfully removed bibliographic content")
        print("   while preserving scientific information!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()

def example_batch_processing():
    """Example of batch processing with LLM preprocessing"""
    
    print("\n" + "=" * 50)
    print("📚 Batch Processing Example")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY required for batch processing")
        return
    
    # Create sample directory with multiple documents
    data_dir = Path("sample_documents")
    data_dir.mkdir(exist_ok=True)
    
    # Sample documents
    documents = {
        "disease_study.txt": """
        AHPND Disease Resistance in Shrimp Breeding Programs
        
        ABSTRACT
        This study evaluates breeding strategies for AHPND resistance in Penaeus vannamei.
        
        METHODS
        Selective breeding was applied over 3 generations using disease challenge tests.
        
        RESULTS
        Genetic gain for survival reached 15% per generation under AHPND challenge.
        
        REFERENCES
        [Multiple references would be here...]
        """,
        
        "nutrition_study.txt": """
        Feed Formulation for Enhanced Immunity in Shrimp
        
        ABSTRACT
        We tested immunostimulant additives in commercial shrimp feeds.
        
        METHODS
        Four experimental diets were tested with 100 shrimp each over 60 days.
        
        RESULTS
        Beta-glucan supplementation improved survival by 25% during disease challenges.
        
        FUNDING
        Supported by the Aquaculture Nutrition Research Grant.
        """
    }
    
    try:
        # Create sample files
        for filename, content in documents.items():
            with open(data_dir / filename, 'w') as f:
                f.write(content)
        
        # Initialize service with LLM preprocessing
        service = DocumentIngestionService(
            enable_llm_preprocessing=True,
            chunking_mode="sentence"
        )
        
        # Process all documents
        processed_docs = service.batch_ingest_from_directory(
            data_dir, 
            pattern="*.txt"
        )
        
        print(f"📊 Processed {len(processed_docs)} documents:")
        
        for doc in processed_docs:
            preprocessing = doc.metadata.get('preprocessing', {})
            original_len = preprocessing.get('original_length', 0)
            cleaned_len = preprocessing.get('cleaned_length', 0)
            reduction = preprocessing.get('reduction_ratio', 0)
            
            print(f"   • {doc.title}")
            print(f"     Sentences: {len(doc.sentences)}")
            print(f"     Text reduction: {reduction:.1%} ({original_len} → {cleaned_len} chars)")
        
        print("\n✅ Batch processing completed with LLM preprocessing")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    finally:
        # Clean up
        if data_dir.exists():
            import shutil
            shutil.rmtree(data_dir)

if __name__ == "__main__":
    example_with_preprocessing()
    example_batch_processing()
#!/usr/bin/env python3
"""
Enhanced Document Extractor CLI
Advanced document processing with OCR, text chunking, and quality filtering
"""
import sys
import argparse
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion.enhanced_document_extractor import EnhancedDocumentExtractor

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Document Extractor - Process PDFs and DOCX files with OCR and intelligent text chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with GUI file selection
  python enhanced_document_extractor_cli.py

  # Process specific directory
  python enhanced_document_extractor_cli.py -i ./documents -o ./output.jsonl

  # Advanced configuration
  python enhanced_document_extractor_cli.py -i ./docs -o ./chunks.jsonl \\
    --max-tokens 1024 --min-chunk-size 200 --overlap 100 \\
    --language en --quality 0.8 --workers 8

  # Disable OCR for faster processing
  python enhanced_document_extractor_cli.py -i ./docs -o ./text.jsonl --no-ocr
        """
    )
    
    # Input/Output options
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        help="Input directory containing PDF and DOCX files (default: GUI selection or ./documents)"
    )
    
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        help="Output JSONL file path (default: GUI selection or ./pretraining_data.jsonl)"
    )
    
    # Text processing options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)"
    )
    
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Minimum chunk size in characters (default: 100)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in characters (default: 50)"
    )
    
    # Quality filtering options
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Target language for filtering (default: en, use 'auto' to disable filtering)"
    )
    
    parser.add_argument(
        "--quality",
        type=float,
        default=0.7,
        help="Minimum text quality score (0.0-1.0, default: 0.7)"
    )
    
    # OCR options
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR processing for faster execution"
    )
    
    # Performance options
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads (default: auto-detect based on CPU cores)"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )

    args = parser.parse_args()
    
    # Configure logging level
    import logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Handle language filtering
    target_language = None if args.language.lower() == 'auto' else args.language
    
    try:
        # Initialize extractor
        extractor = EnhancedDocumentExtractor(
            input_dir=args.input_dir,
            output_file=args.output_file,
            max_tokens=args.max_tokens,
            min_chunk_size=args.min_chunk_size,
            overlap=args.overlap,
            target_language=target_language,
            min_text_quality=args.quality,
            enable_ocr=not args.no_ocr,
            num_workers=args.workers
        )
        
        # Print configuration
        if not args.quiet:
            print(f"\n🔧 Configuration:")
            print(f"   Input Directory: {extractor.input_dir}")
            print(f"   Output File: {extractor.output_file}")
            print(f"   Max Tokens: {extractor.max_tokens}")
            print(f"   Min Chunk Size: {extractor.min_chunk_size}")
            print(f"   Overlap: {extractor.overlap}")
            print(f"   Target Language: {extractor.target_language or 'Any'}")
            print(f"   Quality Threshold: {extractor.min_text_quality}")
            print(f"   OCR Enabled: {extractor.enable_ocr}")
            print(f"   Workers: {extractor.num_workers}")
            print(f"\n🚀 Starting document processing...\n")
        
        # Process documents
        extractor.process_documents()
        
        if not args.quiet:
            print(f"\n✅ Processing complete!")
            print(f"📂 Check output files:")
            print(f"   📝 Text chunks: {extractor.output_file}")
            print(f"   📊 Metadata: {extractor.output_file.with_name(f'{extractor.output_file.stem}_metadata.jsonl')}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

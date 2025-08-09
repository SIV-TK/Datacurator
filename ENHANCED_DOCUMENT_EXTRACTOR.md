# Enhanced Document Extractor

The Enhanced Document Extractor is a sophisticated document processing system that provides advanced OCR capabilities, intelligent text chunking, quality filtering, and language detection.

## Features

### 🔍 **Document Processing**
- **Multi-format Support**: PDF, DOCX, and DOC files
- **OCR Integration**: Extract text from scanned documents and images using Tesseract
- **Parallel Processing**: Multi-threaded processing for faster extraction
- **Resume Capability**: Continue from interrupted processing sessions

### 🧩 **Intelligent Text Chunking**
- **Token-based Chunking**: Uses OpenAI's tiktoken for precise token counting
- **Smart Boundaries**: Respects paragraph and sentence boundaries
- **Configurable Overlap**: Maintains context between chunks
- **Size Control**: Configurable minimum and maximum chunk sizes

### 🏷️ **Quality & Language Control**
- **Language Detection**: Automatic language identification with filtering
- **Quality Scoring**: Text quality assessment based on:
  - Alphabetic character ratio
  - Valid word ratio  
  - Special character density
- **Duplicate Detection**: Content hashing to prevent duplicate chunks
- **Quality Thresholds**: Configurable minimum quality scores

### 📊 **Metadata & Tracking**
- **Processing Statistics**: Detailed stats on files processed, OCR usage, language filtering
- **Document Metadata**: Source file, type, language, OCR status for each chunk
- **Progress Tracking**: Real-time progress bars and status updates
- **Error Handling**: Comprehensive error tracking and recovery

## Usage

### Command Line Interface

```bash
# Basic usage with GUI file selection
python enhanced_document_extractor_cli.py

# Process specific directory
python enhanced_document_extractor_cli.py -i ./documents -o ./output.jsonl

# Advanced configuration
python enhanced_document_extractor_cli.py \
  -i ./docs -o ./chunks.jsonl \
  --max-tokens 1024 --min-chunk-size 200 --overlap 100 \
  --language en --quality 0.8 --workers 8

# Disable OCR for faster processing
python enhanced_document_extractor_cli.py \
  -i ./docs -o ./text.jsonl --no-ocr
```

### Web Interface

The enhanced extractor is integrated into the Data Curator web interface:

1. **Navigate to Process page**
2. **Select "Enhanced Extraction" method**
3. **Configure extraction parameters:**
   - Max tokens per chunk (256-2048)
   - Minimum chunk size (50-1000 chars)
   - Chunk overlap (0-200 chars)
   - Target language filtering
   - Text quality threshold (0.0-1.0)
   - OCR enable/disable
   - Parallel worker count
4. **Upload files or select folder**
5. **Start processing and monitor progress**

### Python API

```python
from src.ingestion.enhanced_document_extractor import EnhancedDocumentExtractor

# Initialize extractor
extractor = EnhancedDocumentExtractor(
    input_dir="./documents",
    output_file="./chunks.jsonl",
    max_tokens=512,
    min_chunk_size=100,
    overlap=50,
    target_language="en",
    min_text_quality=0.7,
    enable_ocr=True,
    num_workers=4
)

# Process documents
extractor.process_documents()
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 512 | Maximum tokens per chunk |
| `min_chunk_size` | 100 | Minimum chunk size in characters |
| `overlap` | 50 | Character overlap between chunks |
| `target_language` | "en" | Target language ("auto" for no filtering) |
| `min_text_quality` | 0.7 | Minimum text quality score (0.0-1.0) |
| `enable_ocr` | True | Enable OCR for image text extraction |
| `num_workers` | auto | Number of parallel processing workers |

## Output Format

### Text Chunks (JSONL)
Each line contains a single text chunk as a JSON string:
```json
"This is the extracted and processed text chunk from the document..."
```

### Metadata (JSONL)
Each line contains metadata for the corresponding document:
```json
{
  "source": "document.pdf",
  "file_type": "pdf", 
  "language": "en",
  "ocr_used": false,
  "status": "processed"
}
```

## Dependencies

```
PyMuPDF>=1.23.12      # PDF processing
pytesseract>=0.3.10   # OCR capabilities  
Pillow>=10.0.0        # Image processing
python-docx>=1.1.0    # DOCX support
tiktoken>=0.5.2       # Token counting
langdetect>=1.0.9     # Language detection
tqdm>=4.66.1          # Progress bars
```

## Performance Considerations

### **Memory Usage**
- Large documents are processed page-by-page to minimize memory usage
- Chunk processing is batched to prevent memory overflow
- Temporary files are cleaned up automatically

### **Processing Speed**
- Parallel processing scales with CPU cores
- OCR can be disabled for faster text-only extraction  
- Resume capability prevents reprocessing completed files
- Efficient regex compilation and text quality algorithms

### **Storage Optimization**
- Duplicate detection prevents redundant chunks
- Quality filtering reduces low-value content
- Compressed output formats supported
- Metadata stored separately for efficient querying

## Integration with Data Curator

The Enhanced Document Extractor is fully integrated with the Data Curator system:

- **Web UI Integration**: Available through the Process page
- **Database Storage**: Results automatically saved to datasets
- **Job Tracking**: Background processing with progress monitoring  
- **API Endpoints**: RESTful API for programmatic access
- **File Management**: Automatic file organization and cleanup

## Troubleshooting

### **Common Issues**

1. **Tesseract not found**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

2. **Memory errors with large PDFs**
   - Reduce `num_workers`
   - Process files individually
   - Increase system memory

3. **Language detection failures**
   - Set `target_language` to "auto"
   - Increase `min_chunk_size`
   - Check document text quality

4. **Low extraction quality**
   - Enable OCR for scanned documents
   - Adjust `min_text_quality` threshold
   - Check source document quality

### **Performance Tuning**

- **CPU-bound**: Increase `num_workers`
- **Memory-limited**: Decrease `num_workers` and `max_tokens`
- **Storage-limited**: Increase `min_text_quality` threshold
- **Speed priority**: Disable OCR and reduce quality thresholds

## Examples

### **Academic Papers**
```bash
python enhanced_document_extractor_cli.py \
  -i ./academic_papers -o ./paper_chunks.jsonl \
  --max-tokens 1024 --quality 0.8 --language en
```

### **Legal Documents**  
```bash
python enhanced_document_extractor_cli.py \
  -i ./legal_docs -o ./legal_chunks.jsonl \
  --max-tokens 512 --overlap 100 --quality 0.9
```

### **Scanned Historical Documents**
```bash
python enhanced_document_extractor_cli.py \
  -i ./scanned_docs -o ./historical_chunks.jsonl \
  --quality 0.6 --workers 2  # OCR enabled by default
```

## License & Credits

Part of the Data Curator project. Uses:
- **PyMuPDF** for PDF processing
- **Tesseract OCR** for text recognition
- **OpenAI tiktoken** for tokenization
- **langdetect** for language identification

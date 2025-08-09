import json
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import langdetect
from langdetect import DetectorFactory
import logging
from docx import Document
import os
import concurrent.futures

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# Try importing tkinter, but allow fallback
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except (ImportError, RuntimeError):
    TKINTER_AVAILABLE = False

# Consistent language detection
DetectorFactory.seed = 0

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class EnhancedDocumentExtractor:
    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        max_tokens: Optional[int] = 512,
        min_chunk_size: int = 100,
        overlap: int = 50,
        target_language: Optional[str] = "en",
        min_text_quality: float = 0.7,
        enable_ocr: bool = True,
        num_workers: int = None
    ):
        # Default directories
        default_input_dir = "./documents"
        default_output_file = "./pretraining_data.jsonl"

        # Handle input directory selection
        if input_dir:
            self.input_dir = Path(input_dir)
        else:
            if TKINTER_AVAILABLE and os.environ.get('DISPLAY'):
                try:
                    root = tk.Tk()
                    root.withdraw()
                    self.input_dir = Path(filedialog.askdirectory(title="Select Input Directory") or default_input_dir)
                    root.destroy()
                except Exception:
                    logging.warning("Falling back to console input for directory selection")
                    self.input_dir = self._prompt_directory("input directory", default_input_dir)
            else:
                self.input_dir = self._prompt_directory("input directory", default_input_dir)

        if not self.input_dir.exists():
            raise ValueError(f"Input directory {self.input_dir} does not exist")

        # Handle output file selection
        if output_file:
            self.output_file = Path(output_file)
        else:
            if TKINTER_AVAILABLE and os.environ.get('DISPLAY'):
                try:
                    root = tk.Tk()
                    root.withdraw()
                    selected_file = filedialog.asksaveasfilename(
                        title="Select Output File",
                        defaultextension=".jsonl",
                        filetypes=[("JSON Lines", "*.jsonl"), ("All files", "*.*")]
                    )
                    self.output_file = Path(selected_file or default_output_file)
                    root.destroy()
                except Exception:
                    logging.warning("Falling back to console input for output file selection")
                    self.output_file = self._prompt_directory("output file", default_output_file, is_file=True)
            else:
                self.output_file = self._prompt_directory("output file", default_output_file, is_file=True)

        self.max_tokens = max_tokens
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.target_language = target_language
        self.min_text_quality = min_text_quality
        self.enable_ocr = enable_ocr
        self.num_workers = num_workers or os.cpu_count() // 2 or 1
        self.seen_hashes = self._load_existing_hashes()
        self._setup_encoder()

    def _setup_encoder(self):
        """Lazy load the tokenizer only when needed"""
        if self.max_tokens:
            import tiktoken
            self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = None

    def _prompt_directory(self, dir_type: str, default: str, is_file: bool = False) -> Path:
        """Prompt user for directory or file path via console with a default option."""
        prompt = f"Enter {dir_type} (default: {default}): "
        user_input = input(prompt).strip()
        path = Path(user_input or default)

        if is_file:
            # Ensure parent directory exists for output file
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Ensure input directory exists
            path.mkdir(parents=True, exist_ok=True)

        return path

    def _load_existing_hashes(self) -> set:
        """Prevent duplicate processing by loading already written hashes from output file."""
        seen = set()
        if self.output_file.exists():
            logging.info("Resuming from previous output...")
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        text = record["text"]  # Now directly accessing the text string
                        hash_ = self.generate_content_hash(text)
                        seen.add(hash_)
                    except Exception:
                        continue
        return seen

    def calculate_text_quality(self, text: str) -> float:
        """Efficient text quality calculation using precomputed values"""
        total_chars = len(text)
        if total_chars == 0:
            return 0.0

        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / total_chars

        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        if word_count == 0:
            return 0.0

        valid_words = sum(1 for word in words if word.isalpha())
        word_ratio = valid_words / word_count

        special_chars = len(re.findall(r'[^\w\s]', text))
        special_ratio = 1 - (special_chars / total_chars)

        return (alpha_ratio * 0.4 + word_ratio * 0.4 + special_ratio * 0.2)

    def detect_language(self, text: str) -> Optional[str]:
        """Efficient language detection using sampling"""
        sample_text = text[:2000]  # Use first 2000 characters for detection
        if len(sample_text.strip()) < 10:
            return None
        try:
            return langdetect.detect(sample_text)
        except:
            return None

    def generate_content_hash(self, text: str) -> str:
        normalized = re.sub(r'\s+', ' ', text).strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        content_hash = self.generate_content_hash(text)
        return content_hash in self.seen_hashes

    def clean_text(self, text: str) -> str:
        """Optimized text cleaning with compiled regex patterns"""
        if not text:
            return ""

        # Compile regex patterns once
        if not hasattr(self, 'clean_patterns'):
            self.clean_patterns = {
                'whitespace': re.compile(r'\s+'),
                'hyphen_newline': re.compile(r'-\n'),
                'newline': re.compile(r'\n'),
                'control_chars': re.compile(r'[\x00-\x1f\x7f-\x9f]'),
                'form_feed': re.compile(r'\x0c'),
                'header_footer': re.compile(r'^\d+\s+\w+\s+\d+$', flags=re.MULTILINE)
            }

        text = self.clean_patterns['hyphen_newline'].sub('', text)
        text = self.clean_patterns['header_footer'].sub('', text)
        text = self.clean_patterns['control_chars'].sub('', text)
        text = self.clean_patterns['form_feed'].sub('', text)
        text = self.clean_patterns['newline'].sub(' ', text)
        text = self.clean_patterns['whitespace'].sub(' ', text).strip()
        return text

    def extract_pdf_text(self, pdf_path: Path) -> Tuple[str, bool]:
        """Extract text from PDF using PyMuPDF with OCR fallback per page"""
        text = ""
        is_ocr = False

        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                # First try text extraction
                page_text = page.get_text()
                if page_text.strip() and len(page_text) > 50:  # Valid text
                    text += self.clean_text(page_text) + "\n"
                elif self.enable_ocr:
                    # Use OCR only for this page
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    cleaned_ocr = self.clean_text(ocr_text)
                    if cleaned_ocr:
                        text += cleaned_ocr + "\n"
                        is_ocr = True
            doc.close()
        except Exception as e:
            logging.warning(f"PDF processing failed for {pdf_path.name}: {e}")
            if self.enable_ocr:
                logging.info(f"Attempting full OCR for {pdf_path.name}")
                try:
                    doc = fitz.open(pdf_path)
                    for page in doc:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        text += self.clean_text(ocr_text) + "\n"
                    is_ocr = True
                except Exception as e2:
                    logging.warning(f"OCR failed for {pdf_path.name}: {e2}")

        return text, is_ocr

    def extract_docx_text(self, docx_path: Path) -> Tuple[str, bool]:
        text = ""
        try:
            doc = Document(docx_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += self.clean_text(para.text) + "\n"
        except Exception as e:
            logging.warning(f"Failed to process {docx_path.name}: {e}")
        return text, False

    def extract_text(self, file_path: Path) -> Tuple[str, bool, str]:
        ext = file_path.suffix.lower()
        if ext == '.pdf':
            text, is_ocr = self.extract_pdf_text(file_path)
            return text, is_ocr, 'pdf'
        elif ext in ('.docx', '.doc'):
            text, is_ocr = self.extract_docx_text(file_path)
            return text, is_ocr, 'docx'
        return "", False, 'unknown'

    def chunk_text(self, text: str) -> List[str]:
        """Efficient text chunking with token-based boundaries"""
        if not self.max_tokens or not text.strip():
            return [text] if text.strip() else []

        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_token_count = 0

        for para in paragraphs:
            para_tokens = self.encoder.encode(para) if self.encoder else []
            para_token_count = len(para_tokens)

            # If paragraph is too big, split into sentences
            if para_token_count > self.max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    sent_tokens = self.encoder.encode(sentence) if self.encoder else []
                    sent_token_count = len(sent_tokens)

                    # Add sentence to current chunk if it fits
                    if current_token_count + sent_token_count <= self.max_tokens:
                        current_chunk.append(sentence)
                        current_token_count += sent_token_count
                    else:
                        # Finalize current chunk
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            if len(chunk_text) >= self.min_chunk_size:
                                chunks.append(chunk_text)

                            # Start new chunk with overlap
                            overlap_sents = current_chunk[-min(len(current_chunk), 3):]  # Last 1-3 sentences
                            current_chunk = overlap_sents + [sentence]
                            current_token_count = sum(len(self.encoder.encode(s)) for s in current_chunk)
                        else:
                            current_chunk = [sentence]
                            current_token_count = sent_token_count
            else:
                # Add entire paragraph to current chunk
                if current_token_count + para_token_count <= self.max_tokens:
                    current_chunk.append(para)
                    current_token_count += para_token_count
                else:
                    # Finalize current chunk
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        if len(chunk_text) >= self.min_chunk_size:
                            chunks.append(chunk_text)

                        # Start new chunk with overlap
                        overlap_paras = [current_chunk[-1]] if current_chunk else []
                        current_chunk = overlap_paras + [para]
                        current_token_count = para_token_count + (len(self.encoder.encode(overlap_paras[0])) if overlap_paras else 0)
                    else:
                        current_chunk = [para]
                        current_token_count = para_token_count

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def process_single_file(self, file_path: Path, stats: Dict[str, int]) -> Tuple[List[str], Dict]:
        """Process a single file and return chunks along with document metadata"""
        doc_metadata = {
            "source": file_path.name,
            "file_type": "unknown",
            "language": None,
            "ocr_used": False,
            "status": "processed"
        }
        try:
            text, is_ocr, file_type = self.extract_text(file_path)
            doc_metadata.update({
                "file_type": file_type,
                "ocr_used": is_ocr
            })

            if not text.strip():
                doc_metadata["status"] = "no_text"
                return [], doc_metadata

            # Update file type stats
            if file_type == 'pdf':
                stats["pdf_files"] += 1
            elif file_type == 'docx':
                stats["docx_files"] += 1

            # Language detection
            lang = None
            if self.target_language:
                lang = self.detect_language(text)
                doc_metadata["language"] = lang
                if lang != self.target_language:
                    stats["language_filtered"] += 1
                    doc_metadata["status"] = "language_filtered"
                    return [], doc_metadata

            # Chunk text
            chunks = self.chunk_text(text)
            filtered_chunks = []

            for chunk in chunks:
                # Skip duplicates and low-quality chunks
                if self.is_duplicate(chunk):
                    stats["duplicates_removed"] += 1
                    continue

                quality_score = self.calculate_text_quality(chunk)
                if quality_score < self.min_text_quality:
                    continue

                filtered_chunks.append(chunk)

                # Update seen hashes
                content_hash = self.generate_content_hash(chunk)
                self.seen_hashes.add(content_hash)

            if filtered_chunks:
                stats["processed"] += 1
                if is_ocr:
                    stats["ocr_used"] += 1
            else:
                doc_metadata["status"] = "no_valid_chunks"

            return filtered_chunks, doc_metadata
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            doc_metadata["status"] = f"error: {str(e)}"
            return [], doc_metadata

    def process_documents(self):
        # Collect files
        pdf_files = list(self.input_dir.rglob("*.pdf"))
        docx_files = list(self.input_dir.rglob("*.docx"))
        files = pdf_files + docx_files
        if not files:
            logging.warning(f"No PDF or Word files found in {self.input_dir}")
            return

        stats = {
            "total_files": len(files),
            "processed": 0,
            "language_filtered": 0,
            "duplicates_removed": 0,
            "ocr_used": 0,
            "pdf_files": 0,
            "docx_files": 0
        }

        # Prepare output files
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file = self.output_file.with_name(f"{self.output_file.stem}_metadata.jsonl")

        # Process files
        with open(self.output_file, 'a', encoding='utf-8') as outfile, \
             open(metadata_file, 'a', encoding='utf-8') as meta_outfile:

            if self.num_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Process files in parallel
                    future_to_file = {
                        executor.submit(self.process_single_file, file, stats): file
                        for file in files
                    }

                    # Create progress bar
                    iterator = tqdm(
                        concurrent.futures.as_completed(future_to_file),
                        total=len(files),
                        desc="Processing Documents"
                    ) if USE_TQDM else concurrent.futures.as_completed(future_to_file)

                    # Collect results
                    for future in iterator:
                        chunks, doc_metadata = future.result()
                        for chunk in chunks:
                            # Write chunk as JSON string directly
                            outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                        meta_outfile.write(json.dumps(doc_metadata, ensure_ascii=False) + '\n')
            else:
                # Sequential processing
                iterator = tqdm(files, desc="Processing Documents") if USE_TQDM else files
                for file in iterator:
                    chunks, doc_metadata = self.process_single_file(file, stats)
                    for chunk in chunks:
                        # Write chunk as JSON string directly
                        outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    meta_outfile.write(json.dumps(doc_metadata, ensure_ascii=False) + '\n')

        logging.info("\n✅ Processing Complete")
        for key, val in stats.items():
            logging.info(f"- {key.replace('_', ' ').capitalize()}: {val}")
        logging.info(f"\n📝 Output saved to: {self.output_file}")
        logging.info(f"\n📄 Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    extractor = EnhancedDocumentExtractor(
        max_tokens=512,
        min_chunk_size=100,
        overlap=50,
        target_language="en",
        min_text_quality=0.65,
        enable_ocr=True,
        num_workers=4  # Optimal for I/O bound tasks
    )
    extractor.process_documents()

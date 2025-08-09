"""
Advanced text cleaning module with grammar correction, entity filtering, and sensitive data removal.
"""
import os
import json
import re
import html
import logging
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import spacy
import ftfy
from langdetect import detect
import language_tool_python

from .config import get_settings

logger = logging.getLogger(__name__)


class AdvancedTextCleaner:
    """Advanced text cleaner with comprehensive filtering and correction capabilities."""
    
    def __init__(self):
        """Initialize the advanced text cleaner."""
        self.config = get_settings()
        
        # Configuration
        self.working_dir = self.config.ADVANCED_CLEANER_WORKING_DIR
        self.target_language = self.config.ADVANCED_CLEANER_TARGET_LANGUAGE
        self.min_words = self.config.ADVANCED_CLEANER_MIN_WORDS
        self.max_special_char_ratio = self.config.ADVANCED_CLEANER_MAX_SPECIAL_CHAR_RATIO
        self.use_spacy = self.config.ADVANCED_CLEANER_USE_SPACY
        self.use_language_tool = self.config.ADVANCED_CLEANER_USE_LANGUAGE_TOOL
        
        # Initialize spaCy model
        self.nlp = None
        self._init_spacy()
        
        # Initialize LanguageTool
        self.language_tool = None
        self._init_language_tool()
        
        # Common brand names and irrelevant entities to remove
        self.excluded_entities = {
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'twitter', 'instagram',
            'tesla', 'netflix', 'adobe', 'ibm', 'intel', 'nvidia', 'coca-cola', 'pepsi',
            'john doe', 'jane doe'  # Common placeholder names
        }
        
        # Regex patterns for sensitive data
        self.email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_regex = re.compile(r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        self.url_regex = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        self.address_regex = re.compile(
            r'\b\d{1,5}\s+[\w\s,.-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b', 
            re.IGNORECASE
        )
        self.excessive_punctuation = re.compile(r'[\W_]{3,}')
        self.boilerplate_phrases = re.compile(
            r'\b(?:read more|click here|sign up|log in|subscribe now|all rights reserved|terms of service|privacy policy|contact us)\b',
            re.IGNORECASE
        )
    
    def _init_spacy(self):
        """Initialize spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _init_language_tool(self):
        """Initialize LanguageTool."""
        # Temporarily disable LanguageTool due to download issues
        logger.warning("LanguageTool disabled due to initialization issues")
        self.language_tool = None
        # try:
        #     self.language_tool = language_tool_python.LanguageTool('en-US')
        #     logger.info("LanguageTool initialized successfully")
        # except Exception as e:
        #     logger.error(f"Failed to initialize LanguageTool: {e}")
        #     self.language_tool = None
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean text by removing sensitive data and noise, keeping meaningful entities,
        and ensuring good English grammar. Returns None if invalid or not English.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text or None if text should be filtered out
        """
        try:
            # Fix encoding issues
            text = ftfy.fix_text(text)
            text = html.unescape(text)
            text = text.replace('\xa0', ' ')

            # Check language
            try:
                if detect(text) != self.target_language:
                    logger.debug("Text not in target language")
                    return None
            except Exception as e:
                logger.debug(f"Language detection failed: {e}")
                return None

            # Remove emails, phone numbers, URLs, addresses, and boilerplate
            text = self.email_regex.sub('', text)
            text = self.phone_regex.sub('', text)
            text = self.url_regex.sub('', text)
            text = self.address_regex.sub('', text)
            text = self.boilerplate_phrases.sub('', text)

            # Remove excessive punctuation
            text = self.excessive_punctuation.sub(' ', text)

            # Normalize whitespace and fix sentence spacing
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text)
            text = text.strip()

            # Apply spaCy for entity recognition, but only remove excluded entities
            if self.nlp:
                try:
                    doc = self.nlp(text)
                    entities_to_remove = []
                    for ent in doc.ents:
                        if ent.text.lower() in self.excluded_entities:
                            entities_to_remove.append((ent.start_char, ent.end_char))
                    for start, end in sorted(entities_to_remove, reverse=True):
                        text = text[:start] + ' ' + text[end:]
                except Exception as e:
                    logger.warning(f"spaCy processing failed: {e}")

            # Remove specific excluded entities (e.g., brand names, placeholder names)
            for entity in self.excluded_entities:
                text = re.sub(r'\b' + re.escape(entity) + r'\b', '', text, flags=re.IGNORECASE)

            # Final cleanup
            text = re.sub(r'\s+', ' ', text).strip()

            # Basic filters
            if len(text.split()) < self.min_words:
                logger.debug("Text too short after cleaning")
                return None
            if sum(1 for c in text if not c.isalnum() and c not in " .,!?'\"") / len(text) > self.max_special_char_ratio:
                logger.debug("Text contains excessive special characters")
                return None
            if text.count("http") > 3 or any(sym in text for sym in ['{', '}', '=', 'function']):
                logger.debug("Text contains code or excessive URLs")
                return None

            # Apply LanguageTool for grammar, spelling, and style corrections
            if self.language_tool:
                try:
                    matches = self.language_tool.check(text)
                    if matches:
                        text = language_tool_python.utils.correct(text, matches)
                        text = re.sub(r'\s+', ' ', text).strip()
                except Exception as e:
                    logger.warning(f"LanguageTool failed: {e}")

            # Final length check
            if len(text.split()) < self.min_words:
                logger.debug("Text too short after LanguageTool corrections")
                return None

            return text
        except Exception as e:
            logger.error(f"Failed to clean text: {e}")
            return None
    
    def clean_texts_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            List of cleaned texts (excludes None results)
        """
        cleaned_texts = []
        
        iterator = tqdm(texts, desc="Cleaning texts") if show_progress else texts
        
        for text in iterator:
            cleaned = self.clean_text(text)
            if cleaned:
                cleaned_texts.append(cleaned)
        
        logger.info(f"Cleaned {len(cleaned_texts)} out of {len(texts)} texts")
        return cleaned_texts
    
    def clean_dataset(self, data: List[Dict[str, Any]], text_field: str = 'text') -> List[Dict[str, Any]]:
        """
        Clean a dataset by applying text cleaning to a specific field.
        
        Args:
            data: List of data records
            text_field: Field name containing text to clean
            
        Returns:
            List of cleaned data records
        """
        cleaned_data = []
        
        for record in tqdm(data, desc="Cleaning dataset"):
            if text_field in record:
                cleaned_text = self.clean_text(record[text_field])
                if cleaned_text:
                    cleaned_record = record.copy()
                    cleaned_record[text_field] = cleaned_text
                    cleaned_record['cleaned_at'] = self._get_timestamp()
                    cleaned_data.append(cleaned_record)
        
        logger.info(f"Cleaned {len(cleaned_data)} out of {len(data)} records")
        return cleaned_data
    
    def load_texts_from_jsonl(self, file_path: str) -> List[str]:
        """Load texts from JSONL file."""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'text' in entry:
                        texts.append(entry['text'])
                    else:
                        logger.warning(f"Skipping entry without 'text' key")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON line: {line.strip()}")
                    continue
        
        logger.info(f"Loaded {len(texts)} text entries from {file_path}")
        return texts
    
    def save_texts_to_jsonl(self, texts: List[str], output_path: str):
        """Save cleaned texts to JSONL file."""
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if dirname is not empty
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(texts)} texts to {output_path}")
    
    def save_text_incremental(self, text: str, output_path: str):
        """Save a single cleaned text entry to the output file incrementally."""
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if dirname is not empty
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    
    def run_cleaning_pipeline(self, input_path: str, output_path: str) -> int:
        """
        Run the complete text cleaning pipeline.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            
        Returns:
            Number of texts successfully cleaned
        """
        try:
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            texts = self.load_texts_from_jsonl(input_path)
            
            logger.info(f"Starting cleaning pipeline for {len(texts)} texts")
            cleaned_count = 0
            
            # Process texts incrementally to avoid memory issues
            for text in tqdm(texts, desc="Cleaning texts"):
                cleaned = self.clean_text(text)
                if cleaned:
                    self.save_text_incremental(cleaned, output_path)
                    cleaned_count += 1

            logger.info(f"Pipeline completed: {cleaned_count} texts cleaned and saved")
            
            if cleaned_count < len(texts):
                skipped = len(texts) - cleaned_count
                logger.warning(f"Skipped {skipped} entries due to cleaning filters")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cleaning pipeline failed: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.language_tool:
            try:
                self.language_tool.close()
            except:
                pass
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def prompt_for_input_file():
    """Prompt user for input file path."""
    return input("📥 Enter path to input JSONL file with text: ").strip()


def run_cleaning_pipeline():
    """
    Standalone function to run the cleaning pipeline with user prompts.
    """
    try:
        input_path = prompt_for_input_file()
        if not os.path.isfile(input_path):
            print("❌ Input file not found.")
            return

        working_dir = "data"
        output_file = os.path.join(working_dir, "cleaned_texts.jsonl")
        
        with AdvancedTextCleaner() as cleaner:
            cleaned_count = cleaner.run_cleaning_pipeline(input_path, output_file)
            
        print(f"✅ Cleaned and saved {cleaned_count} text entries to: {output_file}")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    run_cleaning_pipeline()

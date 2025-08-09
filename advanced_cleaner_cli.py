#!/usr/bin/env python3
"""
Standalone advanced text cleaning script with grammar correction, entity filtering, and sensitive data removal.
"""
import os
import json
import re
import html
import logging
from tqdm import tqdm
import spacy
import ftfy
from langdetect import detect
import language_tool_python

# ----------------- CONFIG -----------------
WORKING_DIR = "data"
OUTPUT_FILE = os.path.join(WORKING_DIR, "cleaned_texts.jsonl")
LOG_FILE = os.path.join(WORKING_DIR, "cleaner.log")
TARGET_LANGUAGE = "en"

# Logging configuration
os.makedirs(WORKING_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except OSError:
    print("❌ spaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
    raise

# Initialize LanguageTool
try:
    tool = language_tool_python.LanguageTool('en-US')
    print("✅ LanguageTool initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LanguageTool: {e}")
    raise

# Common brand names and irrelevant entities to remove (extend as needed)
EXCLUDED_ENTITIES = {
    'google', 'microsoft', 'apple', 'amazon', 'facebook', 'twitter', 'instagram',
    'tesla', 'netflix', 'adobe', 'ibm', 'intel', 'nvidia', 'coca-cola', 'pepsi',
    'john doe', 'jane doe'  # Common placeholder names
}

# Regex patterns for sensitive data
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
ADDRESS_REGEX = re.compile(r'\b\d{1,5}\s+[\w\s,.-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b', re.IGNORECASE)
EXCESSIVE_PUNCTUATION = re.compile(r'[\W_]{3,}')
BOILERPLATE_PHRASES = re.compile(
    r'\b(?:read more|click here|sign up|log in|subscribe now|all rights reserved|terms of service|privacy policy|contact us)\b',
    re.IGNORECASE
)

def prompt_for_input_file():
    """Prompt user for input file path."""
    return input("📥 Enter path to input JSONL file with text: ").strip()

def load_texts(jsonl_path):
    """Load texts from JSONL file."""
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'text' in entry:
                    texts.append(entry['text'])
                else:
                    logging.warning(f"Skipping entry without 'text' key: {line.strip()}")
            except json.JSONDecodeError:
                logging.error(f"❌ Invalid JSON line: {line.strip()}")
                continue
    print(f"✅ Loaded {len(texts)} text entries.")
    return texts

def clean_text(text):
    """
    Clean text by removing sensitive data and noise, keeping meaningful entities,
    and ensuring good English grammar. Returns None if invalid or not English.
    """
    try:
        # Fix encoding issues
        text = ftfy.fix_text(text)
        text = html.unescape(text)
        text = text.replace('\xa0', ' ')

        # Check language
        try:
            if detect(text) != TARGET_LANGUAGE:
                logging.info("Text not in English")
                return None
        except Exception as e:
            logging.info(f"Language detection failed: {e}")
            return None

        # Remove emails, phone numbers, URLs, addresses, and boilerplate
        text = EMAIL_REGEX.sub('', text)
        text = PHONE_REGEX.sub('', text)
        text = URL_REGEX.sub('', text)
        text = ADDRESS_REGEX.sub('', text)
        text = BOILERPLATE_PHRASES.sub('', text)

        # Remove excessive punctuation
        text = EXCESSIVE_PUNCTUATION.sub(' ', text)

        # Normalize whitespace and fix sentence spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text)
        text = text.strip()

        # Apply spaCy for entity recognition, but only remove excluded entities
        try:
            doc = nlp(text)
            entities_to_remove = []
            for ent in doc.ents:
                if ent.text.lower() in EXCLUDED_ENTITIES:
                    entities_to_remove.append((ent.start_char, ent.end_char))
            for start, end in sorted(entities_to_remove, reverse=True):
                text = text[:start] + ' ' + text[end:]
        except Exception as e:
            logging.warning(f"spaCy processing failed: {e}")

        # Remove specific excluded entities (e.g., brand names, placeholder names)
        for entity in EXCLUDED_ENTITIES:
            text = re.sub(r'\b' + re.escape(entity) + r'\b', '', text, flags=re.IGNORECASE)

        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()

        # Basic filters
        if len(text.split()) < 30:  # Lowered to 30 to retain more meaningful text
            logging.info("Text too short after cleaning")
            return None
        if sum(1 for c in text if not c.isalnum() and c not in " .,!?'\"") / len(text) > 0.1:
            logging.info("Text contains excessive special characters")
            return None
        if text.count("http") > 3 or any(sym in text for sym in ['{', '}', '=', 'function']):
            logging.info("Text contains code or excessive URLs")
            return None

        # Apply LanguageTool for grammar, spelling, and style corrections
        try:
            matches = tool.check(text)
            if matches:
                text = language_tool_python.utils.correct(text, matches)
                text = re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logging.warning(f"LanguageTool failed: {e}")

        # Final length check
        if len(text.split()) < 30:
            logging.info("Text too short after LanguageTool corrections")
            return None

        return text
    except Exception as e:
        logging.error(f"Failed to clean text: {e}")
        return None

def save_cleaned_text(text, output_path):
    """
    Save a single cleaned text entry to the output file.
    """
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

def run_cleaning_pipeline():
    """
    Main pipeline to clean text from a JSONL file.
    """
    try:
        input_path = prompt_for_input_file()
        if not os.path.isfile(input_path):
            print("❌ Input file not found.")
            return

        os.makedirs(WORKING_DIR, exist_ok=True)
        texts = load_texts(input_path)

        print(f"🧹 Cleaning {len(texts)} text entries...")
        cleaned_count = 0
        with tqdm(total=len(texts), desc="Cleaning texts") as pbar:
            for text in texts:
                cleaned = clean_text(text)
                if cleaned:
                    save_cleaned_text(cleaned, OUTPUT_FILE)
                    cleaned_count += 1
                pbar.update(1)

        print(f"✅ Cleaned and saved {cleaned_count} text entries to: {OUTPUT_FILE}")
        if cleaned_count < len(texts):
            print(f"⚠️ Skipped {len(texts) - cleaned_count} entries due to cleaning filters.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print(f"❌ An error occurred. Check {LOG_FILE} for details.")
    finally:
        try:
            tool.close()
        except:
            pass

if __name__ == "__main__":
    run_cleaning_pipeline()

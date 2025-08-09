"""
Data cleaning and processing utilities for LLM training data.
"""
import re
import hashlib
import unicodedata
import html
from typing import List, Dict, Any, Optional, Union, Generator, Set, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException
from bs4 import BeautifulSoup
from loguru import logger
import string
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from transformers import pipeline

# Try to download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK resources: {e}")

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Could not load spaCy model: {e}")
    nlp = None

from ..core.config import get_settings

settings = get_settings()


class TextCleaner:
    """Text cleaning utilities for LLM training data."""
    
    def __init__(
        self,
        remove_html: bool = True,
        fix_encoding: bool = True,
        normalize_whitespace: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        language_filter: Optional[List[str]] = None,
    ):
        """
        Initialize the text cleaner.
        
        Args:
            remove_html: Whether to remove HTML tags
            fix_encoding: Whether to fix encoding issues
            normalize_whitespace: Whether to normalize whitespace
            min_length: Minimum text length (in characters)
            max_length: Maximum text length (in characters)
            language_filter: List of allowed language codes
        """
        self.remove_html = remove_html
        self.fix_encoding = fix_encoding
        self.normalize_whitespace = normalize_whitespace
        self.min_length = min_length
        self.max_length = max_length
        self.language_filter = language_filter or settings.LANGUAGES
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for LLM training.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix encoding issues
        if self.fix_encoding:
            text = self._fix_encoding(text)
        
        # Remove HTML
        if self.remove_html:
            text = self._remove_html(text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text
    
    def _fix_encoding(self, text: str) -> str:
        """
        Fix common encoding issues.
        
        Args:
            text: Input text
            
        Returns:
            Text with fixed encoding
        """
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Replace common problematic characters
        replacements = {
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "--", # Em dash
            "\u00a0": " ",  # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _remove_html(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without HTML tags
        """
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text(separator=' ')
        except Exception as e:
            logger.error(f"Error removing HTML: {e}")
            # Fallback to regex
            return re.sub(r'<[^>]+>', ' ', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def is_valid(self, text: str) -> bool:
        """
        Check if text is valid for LLM training.
        
        Args:
            text: Input text
            
        Returns:
            Whether the text is valid
        """
        if not text:
            return False
        
        # Check length
        if self.min_length and len(text) < self.min_length:
            return False
        
        if self.max_length and len(text) > self.max_length:
            return False
        
        # Check language
        if self.language_filter:
            try:
                lang = detect(text)
                if lang not in self.language_filter:
                    return False
            except LangDetectException:
                # If language detection fails, assume it's invalid
                return False
        
        return True
    
    def calculate_quality_score(self, text: str) -> float:
        """
        Calculate quality score for text.
        
        Args:
            text: Input text
            
        Returns:
            Quality score (0-1)
        """
        if not text:
            return 0.0
        
        # Initialize score
        score = 1.0
        
        # Penalize very short or very long text
        length = len(text)
        if length < 50:
            score *= 0.5
        elif length < 100:
            score *= 0.8
        elif length > 10000:
            score *= 0.7
        
        # Penalize text with many special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / length
        if special_char_ratio > 0.2:
            score *= 0.7
        
        # Penalize text with many numbers
        number_ratio = len(re.findall(r'\d', text)) / length
        if number_ratio > 0.2:
            score *= 0.8
        
        # Penalize text with few sentences
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 3:
                score *= 0.9
        except Exception:
            pass
        
        # Cap score between 0 and 1
        return max(0.0, min(1.0, score))
    
    def get_text_hash(self, text: str) -> str:
        """
        Get hash of text for deduplication.
        
        Args:
            text: Input text
            
        Returns:
            MD5 hash of text
        """
        # Normalize text before hashing
        normalized = self._normalize_for_hash(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _normalize_for_hash(self, text: str) -> str:
        """
        Normalize text for consistent hashing.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove all whitespace
        text = re.sub(r'\s+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text


class DatasetCleaner:
    """Dataset-level cleaning utilities."""
    
    def __init__(
        self,
        text_cleaner: Optional[TextCleaner] = None,
        deduplicate: bool = True,
        use_ai_filter: bool = False,
        ai_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the dataset cleaner.
        
        Args:
            text_cleaner: Text cleaner instance
            deduplicate: Whether to deduplicate data
            use_ai_filter: Whether to use AI-based filtering
            ai_config: Configuration for AI filtering
        """
        self.text_cleaner = text_cleaner or TextCleaner()
        self.deduplicate = deduplicate
        self.use_ai_filter = use_ai_filter
        self.ai_config = ai_config or {}
        
        # Initialize AI pipeline if needed
        self.ai_pipeline = None
        if self.use_ai_filter:
            self._init_ai_pipeline()
    
    def _init_ai_pipeline(self):
        """Initialize AI pipeline for filtering."""
        try:
            model_name = self.ai_config.get('model_name', 'distilbert-base-uncased')
            task = self.ai_config.get('task', 'text-classification')
            
            self.ai_pipeline = pipeline(task, model=model_name)
            logger.info(f"AI pipeline initialized with model {model_name}")
        except Exception as e:
            logger.error(f"Error initializing AI pipeline: {e}")
            self.use_ai_filter = False
    
    def clean_dataset(
        self,
        data: List[Dict[str, Any]],
        content_field: str = 'content',
        output_field: str = 'processed_content',
    ) -> List[Dict[str, Any]]:
        """
        Clean a dataset for LLM training.
        
        Args:
            data: Input dataset
            content_field: Field containing text content
            output_field: Field to store processed content
            
        Returns:
            Cleaned dataset
        """
        # Clean text
        for item in data:
            if content_field in item and item[content_field]:
                text = item[content_field]
                
                # Apply text cleaning
                cleaned_text = self.text_cleaner.clean_text(text)
                
                # Check if text is valid
                is_valid = self.text_cleaner.is_valid(cleaned_text)
                
                # Calculate quality score
                quality_score = self.text_cleaner.calculate_quality_score(cleaned_text)
                
                # Store results
                item[output_field] = cleaned_text
                item['is_valid'] = is_valid
                item['quality_score'] = quality_score
                item['text_hash'] = self.text_cleaner.get_text_hash(cleaned_text)
        
        # Apply AI filtering if enabled
        if self.use_ai_filter and self.ai_pipeline:
            data = self._apply_ai_filter(data, output_field)
        
        # Deduplicate if enabled
        if self.deduplicate:
            data = self._deduplicate(data)
        
        return data
    
    def _apply_ai_filter(
        self,
        data: List[Dict[str, Any]],
        content_field: str = 'processed_content',
    ) -> List[Dict[str, Any]]:
        """
        Apply AI-based filtering to dataset.
        
        Args:
            data: Input dataset
            content_field: Field containing text content
            
        Returns:
            Filtered dataset
        """
        if not self.ai_pipeline:
            return data
        
        try:
            # Get threshold from config
            threshold = self.ai_config.get('threshold', 0.5)
            positive_label = self.ai_config.get('positive_label', 'POSITIVE')
            
            # Apply AI model to each item
            for item in data:
                if content_field in item and item[content_field]:
                    text = item[content_field]
                    
                    # Skip very short texts
                    if len(text) < 20:
                        item['ai_score'] = 0.0
                        continue
                    
                    # Apply AI model
                    result = self.ai_pipeline(text[:1024])  # Truncate to avoid model limits
                    
                    # Extract score
                    if isinstance(result, list) and result:
                        # Find score for positive class
                        for label_info in result:
                            if label_info['label'] == positive_label:
                                item['ai_score'] = label_info['score']
                                break
                        else:
                            item['ai_score'] = 0.0
                    else:
                        item['ai_score'] = 0.0
                    
                    # Update validity based on AI score
                    if item['ai_score'] < threshold:
                        item['is_valid'] = False
        
        except Exception as e:
            logger.error(f"Error applying AI filter: {e}")
        
        return data
    
    def _deduplicate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate dataset based on text hash.
        
        Args:
            data: Input dataset
            
        Returns:
            Deduplicated dataset
        """
        seen_hashes = set()
        deduplicated = []
        
        for item in data:
            if 'text_hash' in item and item['text_hash']:
                if item['text_hash'] not in seen_hashes:
                    seen_hashes.add(item['text_hash'])
                    deduplicated.append(item)
                    item['is_duplicate'] = False
                else:
                    item['is_duplicate'] = True
            else:
                deduplicated.append(item)
                item['is_duplicate'] = False
        
        return deduplicated
    
    def analyze_dataset(
        self,
        data: List[Dict[str, Any]],
        content_field: str = 'processed_content',
    ) -> Dict[str, Any]:
        """
        Analyze dataset and provide statistics.
        
        Args:
            data: Input dataset
            content_field: Field containing text content
            
        Returns:
            Dataset statistics
        """
        stats = {
            'total_records': len(data),
            'valid_records': sum(1 for item in data if item.get('is_valid', False)),
            'invalid_records': sum(1 for item in data if not item.get('is_valid', True)),
            'duplicate_records': sum(1 for item in data if item.get('is_duplicate', False)),
            'avg_quality_score': np.mean([item.get('quality_score', 0) for item in data if 'quality_score' in item]),
            'language_distribution': {},
            'length_distribution': {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
            },
            'quality_distribution': {
                '0.0-0.2': 0,
                '0.2-0.4': 0,
                '0.4-0.6': 0,
                '0.6-0.8': 0,
                '0.8-1.0': 0,
            },
        }
        
        # Collect length and language statistics
        lengths = []
        languages = Counter()
        
        for item in data:
            if content_field in item and item[content_field]:
                text = item[content_field]
                
                # Length stats
                length = len(text)
                lengths.append(length)
                
                # Language detection
                try:
                    lang = detect(text[:1000])  # Use first 1000 chars for performance
                    languages[lang] += 1
                except LangDetectException:
                    languages['unknown'] += 1
                
                # Quality distribution
                score = item.get('quality_score', 0)
                if score < 0.2:
                    stats['quality_distribution']['0.0-0.2'] += 1
                elif score < 0.4:
                    stats['quality_distribution']['0.2-0.4'] += 1
                elif score < 0.6:
                    stats['quality_distribution']['0.4-0.6'] += 1
                elif score < 0.8:
                    stats['quality_distribution']['0.6-0.8'] += 1
                else:
                    stats['quality_distribution']['0.8-1.0'] += 1
        
        # Compute length statistics
        if lengths:
            stats['length_distribution']['min'] = min(lengths)
            stats['length_distribution']['max'] = max(lengths)
            stats['length_distribution']['mean'] = np.mean(lengths)
            stats['length_distribution']['median'] = np.median(lengths)
        
        # Language distribution
        stats['language_distribution'] = {k: v for k, v in languages.most_common()}
        
        return stats


def clean_dataset(
    data: List[Dict[str, Any]],
    content_field: str = 'content',
    output_field: str = 'processed_content',
    remove_html: bool = True,
    fix_encoding: bool = True,
    normalize_whitespace: bool = True,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    language_filter: Optional[List[str]] = None,
    deduplicate: bool = True,
    use_ai_filter: bool = False,
    ai_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Clean a dataset for LLM training.
    
    Args:
        data: Input dataset
        content_field: Field containing text content
        output_field: Field to store processed content
        remove_html: Whether to remove HTML tags
        fix_encoding: Whether to fix encoding issues
        normalize_whitespace: Whether to normalize whitespace
        min_length: Minimum text length (in characters)
        max_length: Maximum text length (in characters)
        language_filter: List of allowed language codes
        deduplicate: Whether to deduplicate data
        use_ai_filter: Whether to use AI-based filtering
        ai_config: Configuration for AI filtering
        
    Returns:
        Tuple of (cleaned dataset, statistics)
    """
    # Initialize cleaners
    text_cleaner = TextCleaner(
        remove_html=remove_html,
        fix_encoding=fix_encoding,
        normalize_whitespace=normalize_whitespace,
        min_length=min_length,
        max_length=max_length,
        language_filter=language_filter,
    )
    
    dataset_cleaner = DatasetCleaner(
        text_cleaner=text_cleaner,
        deduplicate=deduplicate,
        use_ai_filter=use_ai_filter,
        ai_config=ai_config,
    )
    
    # Clean dataset
    cleaned_data = dataset_cleaner.clean_dataset(
        data=data,
        content_field=content_field,
        output_field=output_field,
    )
    
    # Analyze dataset
    stats = dataset_cleaner.analyze_dataset(
        data=cleaned_data,
        content_field=output_field,
    )
    
    return cleaned_data, stats

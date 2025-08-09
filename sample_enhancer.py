import os
import json
from datetime import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def enhance_data(input_file, output_file=None):
    """
    Enhance sample data with NLP processing.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional)
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_enhanced{ext}"
    
    # Read data from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Perform enhancement operations
    for item in data:
        # Get content text
        text = item.get('content', '')
        
        # Skip empty content
        if not text:
            continue
        
        # Basic text statistics
        item['stats'] = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
        }
        
        # Extract keywords (simple approach)
        item['keywords'] = extract_keywords(text)
        
        # Generate summary (simple approach)
        item['summary'] = generate_summary(text)
        
        # Add metadata
        item['enhanced'] = True
        item['enhanced_at'] = datetime.now().isoformat()
    
    # Write enhanced data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Enhanced data saved to {output_file}")
    print(f"Enhanced {len(data)} records")
    
    return output_file


def extract_keywords(text, max_keywords=5):
    """
    Extract keywords from text using a simple frequency-based approach.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequency
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [word for word, freq in sorted_words[:max_keywords]]


def generate_summary(text, max_sentences=1):
    """
    Generate a simple summary by extracting the first few sentences.
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences in the summary
        
    Returns:
        Summary text
    """
    sentences = sent_tokenize(text)
    
    # Return first few sentences
    summary = ' '.join(sentences[:max_sentences])
    
    return summary


if __name__ == "__main__":
    # Enhance the cleaned data
    input_file = os.path.join(os.getcwd(), "data", "sample_data_cleaned.json")
    enhance_data(input_file)

# Advanced Web Scraper & Text Cleaner Integration

The Data Curator system has been redesigned to include both a powerful advanced web scraper and an advanced text cleaner with grammar correction, entity filtering, and sensitive data removal.

## New Features

### 🚀 Advanced Web Scraper (`src/ingestion/advanced_web_scraper.py`)

The new `AdvancedWebScraper` class provides:

- **Multiple Extraction Methods**: 
  - Newspaper3k for article extraction
  - Selenium WebDriver for JavaScript-heavy sites
  - Playwright for modern web applications
  - BeautifulSoup for simple HTML parsing

- **Intelligent Fallbacks**: If one method fails, the scraper automatically tries the next method

- **Concurrent Processing**: Configurable thread pool for parallel scraping

- **Quality Filtering**: 
  - Language detection
  - Content length validation
  - Character distribution analysis
  - HTML/JavaScript code filtering

- **Smart URL Management**:
  - Domain-based filtering
  - File extension exclusion
  - Sub-URL discovery and extraction
  - Duplicate URL prevention

### 🧹 Advanced Text Cleaner (`src/core/advanced_cleaner.py`)

The new `AdvancedTextCleaner` class provides:

- **Grammar & Spelling Correction**: 
  - LanguageTool integration for grammar correction
  - Spelling error detection and correction
  - Style improvements

- **Sensitive Data Removal**:
  - Email addresses
  - Phone numbers
  - Physical addresses
  - URLs and web links

- **Entity Filtering**:
  - spaCy NER for entity recognition
  - Brand name filtering
  - Placeholder name removal
  - Configurable entity exclusion lists

- **Text Quality Enhancement**:
  - Character encoding normalization (ftfy)
  - HTML entity decoding
  - Whitespace normalization
  - Language detection and filtering

- **Boilerplate Removal**:
  - Common website boilerplate phrases
  - Navigation text
  - Copyright notices
  - Call-to-action buttons

### 🛠️ Configuration

Advanced system settings in `config/default.yaml`:

```yaml
# Advanced scraper settings
scraper:
  num_threads: 8                    # Number of concurrent threads
  use_selenium_fallback: true       # Enable Selenium fallback
  rate_limit_delay: 0.5            # Delay between requests (seconds)
  target_language: en              # Target language for content
  max_sub_urls: 100                # Max sub-URLs per main URL
  working_dir: data                # Working directory
  exclude_domains: [...]           # Domains to exclude
  exclude_extensions: [...]        # File extensions to skip

# Advanced text cleaner settings
advanced_cleaner:
  working_dir: data
  target_language: en
  min_words: 30
  max_special_char_ratio: 0.1
  use_spacy: true
  use_language_tool: true
  excluded_entities: [...]         # Entities to filter out
```

## Usage Examples

### 1. Command Line Interface

#### Advanced scraping:
```bash
python -m src.main scrape https://example.com --advanced --output results.json
```

#### Advanced text cleaning:
```bash
python -m src.main clean-advanced input.jsonl --output cleaned.jsonl
```

#### Combined text cleaning with options:
```bash
python -m src.main clean-texts input.jsonl --output cleaned.jsonl --advanced
```

#### Pipeline scraping from file:
```bash
python -m src.main scrape-file example_urls.jsonl --output extracted_texts.jsonl
```

### 2. Standalone CLI Scripts

#### Run the advanced scraper pipeline:
```bash
python advanced_scraper_cli.py
```

#### Run the advanced text cleaner:
```bash
python advanced_cleaner_cli.py
```

### 3. Programmatic Usage

#### Advanced Web Scraping:
```python
from src.ingestion.advanced_web_scraper import AdvancedWebScraper

async with AdvancedWebScraper() as scraper:
    # Single URL with multiple fallbacks
    result, sub_urls = scraper.scrape_article("https://example.com")
    
    # Batch processing
    urls = [{"url": "https://example.com"}, {"url": "https://test.com"}]
    results = scraper.scrape_urls_batch(urls, "output.jsonl")
```

#### Advanced Text Cleaning:
```python
from src.core.advanced_cleaner import AdvancedTextCleaner

with AdvancedTextCleaner() as cleaner:
    # Single text cleaning
    cleaned_text = cleaner.clean_text("Raw text with errors...")
    
    # Batch processing
    cleaned_texts = cleaner.clean_texts_batch(text_list)
    
    # Dataset cleaning
    cleaned_data = cleaner.clean_dataset(dataset, text_field='content')
```

#### Integrated Cleaning in Standard Pipeline:
```python
from src.core.cleaner import TextCleaner

# Enable advanced cleaning in standard cleaner
cleaner = TextCleaner(use_advanced_cleaning=True)
cleaned_text = cleaner.clean_text("Text to clean...")

# Or use advanced cleaning method directly
cleaned_text = cleaner.clean_text_advanced("Text to clean...")
```

### 4. Web API

#### Advanced scraping endpoint:
```javascript
// POST to /api/scrape
{
  "urls": ["https://example.com"],
  "selectors": {"title": "h1", "content": ".content"},
  "follow_links": true,
  "max_pages": 50
}
```

#### Advanced cleaning endpoint:
```javascript
// POST to /api/clean
{
  "dataset_id": 123,
  "use_advanced": true,
  "remove_html": true,
  "min_length": 50,
  "max_length": 10000
}
```

## Dependencies

New dependencies added to `requirements.txt`:
- `newspaper3k==0.2.8` - Article extraction
- `trafilatura==1.6.3` - Text extraction
- `validators==0.22.0` - URL validation
- `aiohttp==3.9.0` - Async HTTP client
- `ftfy==6.1.1` - Text encoding fixes
- `language-tool-python==2.7.1` - Grammar correction

## Quality Assurance

### Web Scraping Quality Filters
1. **Language Detection**: Only extracts content in the target language
2. **Length Validation**: Minimum 100 words required
3. **Character Analysis**: Filters out code-like content
4. **HTML Detection**: Removes pages with excessive HTML/JavaScript
5. **Duplicate Prevention**: Tracks processed URLs to avoid re-scraping

### Text Cleaning Quality Enhancement
1. **Grammar Correction**: LanguageTool fixes grammar and spelling errors
2. **Encoding Normalization**: ftfy fixes character encoding issues
3. **Entity Filtering**: Removes brand names and placeholder entities
4. **Sensitive Data Removal**: Strips emails, phones, addresses
5. **Boilerplate Removal**: Eliminates common website boilerplate text
6. **Language Validation**: Ensures text is in target language

## Performance Features

### Scraping Performance
- **Connection Pooling**: Reuses HTTP connections and browser instances
- **Rate Limiting**: Configurable delays to respect server resources
- **Memory Management**: Streams data to disk to prevent memory overflow
- **Progress Tracking**: Real-time progress bars for long-running operations
- **Error Recovery**: Graceful handling of network errors and timeouts

### Cleaning Performance
- **Batch Processing**: Efficiently processes multiple texts
- **Incremental Saving**: Saves progress to prevent data loss
- **Memory Optimization**: Processes texts individually to manage memory
- **Resource Cleanup**: Properly closes LanguageTool and spaCy resources

## Web Interface Integration

The web interface now includes:

### Advanced Scraping Options
- Toggle between basic and advanced scraping
- Real-time job monitoring
- Progress tracking with detailed results
- Error handling with fallback options

### Advanced Cleaning Options  
- **Advanced Cleaning Toggle**: Enable grammar correction and entity filtering
- **Cleaning Configuration**: Customize minimum/maximum text length
- **Real-time Progress**: Monitor cleaning jobs with live updates
- **Results Preview**: View sample cleaned texts before completion
- **Success Metrics**: Track cleaning success rates and statistics

### Enhanced User Experience
- **Job Queue Management**: Track multiple running jobs
- **Detailed Results**: View comprehensive cleaning and scraping statistics
- **Error Recovery**: Graceful fallback to basic methods when advanced features fail
- **Progress Indicators**: Real-time progress bars and status updates

## Architecture Integration

The advanced features integrate seamlessly with the existing Data Curator architecture:

- **Unified Configuration**: Both systems use the same YAML configuration
- **Database Integration**: Results stored in the same database schema
- **Web Interface**: Accessible through enhanced web UI
- **CLI Integration**: Available through expanded CLI commands
- **Pipeline Compatibility**: Works with existing processing pipelines

## Migration Guide

### From Legacy Scraper
- Legacy `WebScraper` class automatically redirects to `AdvancedWebScraper`
- Existing API endpoints remain compatible
- Configuration keys are backward compatible

### Adding Advanced Cleaning
- Update existing `TextCleaner` instances with `use_advanced_cleaning=True`
- Or use `AdvancedTextCleaner` directly for full control
- Web interface automatically detects and uses advanced features

This comprehensive redesign significantly enhances both data collection and cleaning capabilities while maintaining system modularity and ease of use.

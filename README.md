# Data Curator

A comprehensive data curation tool for cleaning, processing, and enhancing data for training Large Language Models (LLMs).

## Features

- **Data Ingestion**: 
  - Web scraping with support for JavaScript-heavy sites
  - File processing with support for various formats (CSV, JSON, JSONL, XML, etc.)
  - Structured data extraction from various sources

- **Data Cleaning**:
  - HTML removal and text extraction
  - Encoding issues fixing
  - Language filtering
  - Deduplication
  - Quality scoring
  - Automated filtering

- **AI-Powered Enhancement**:
  - Text cleaning and normalization
  - Content rewriting and improvement
  - Metadata extraction
  - Quality assessment
  - Data augmentation

- **Data Management**:
  - SQLAlchemy-based ORM for data tracking
  - Dataset versioning
  - Processing job tracking
  - Quality statistics

## Architecture

The project is structured as follows:

```
data-curator/
├── config/                # Configuration files
├── src/                   # Source code
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration management
│   │   ├── database.py    # Database connection utilities
│   │   ├── logging.py     # Logging configuration
│   │   ├── cleaner.py     # Data cleaning utilities
│   │   └── ai_enhancer.py # AI-based enhancement
│   ├── ingestion/         # Data ingestion modules
│   │   ├── web_scraper.py # Web scraping utilities
│   │   └── file_processor.py # File processing utilities
│   ├── models/            # Data models
│   │   ├── database.py    # SQLAlchemy models
│   │   └── api.py         # API request/response models
│   └── main.py            # Application entry point
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-curator.git
   cd data-curator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   DATABASE_DRIVER=sqlite
   APP_ENV=development
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Command Line Interface

The tool provides a command-line interface for common operations:

#### Scraping websites:

```bash
python -m src.main scrape https://example.com --selector "title:.title" --selector "content:.content" --output data.json
```

#### Processing files:

```bash
python -m src.main process data/input.csv data/input2.json --recursive --output processed_data.json
```

### Sample Workflow

For demonstration purposes, you can use the provided sample scripts:

1. Process sample data:
   ```bash
   python sample_processor.py
   ```

2. Enhance data with NLP features:
   ```bash
   python sample_enhancer.py
   ```

## Known Issues

- The web scraper module (web_scraper.py) is currently having compatibility issues with Pydantic v2
- Some additional dependencies may need to be installed manually
- Database models need to be updated for newer SQLAlchemy versions

## Next Steps

1. Update web scraper to be fully compatible with Pydantic v2
2. Implement more advanced AI enhancement features using OpenAI or Hugging Face models
3. Add data export functionality for various formats
4. Create a web UI for data management

## License

MIT

#### Cleaning data:

```bash
python -m src.main clean data.json --output clean_data.json --remove-html --deduplicate --min-length 100
```

### Python API

You can also use the tool programmatically:

```python
from src.core.config import get_settings, ensure_directories
from src.ingestion.web_scraper import scrape_website
from src.models.api import WebScrapingConfig

# Initialize settings
ensure_directories()
settings = get_settings()

# Configure web scraping
config = WebScrapingConfig(
    urls=["https://example.com"],
    selectors={"title": "h1", "content": "article"},
    follow_links=True,
    max_pages=5
)

# Scrape website
data = scrape_website(config)
print(f"Collected {len(data)} items")
```

## AI Enhancement

To use AI-powered enhancement features, set your API key in the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
```

Then use the enhancer:

```python
from src.core.ai_enhancer import enhance_dataset

enhanced_data = enhance_dataset(
    data=my_data,
    content_field="content",
    output_field="enhanced_content",
    task="clean"
)
```

## Database Schema

The system uses SQLAlchemy ORM with the following main models:

- **Dataset**: Track collections of data
- **DataRecord**: Individual data records
- **Annotation**: Labels and annotations for records
- **ProcessingJob**: Track data processing operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

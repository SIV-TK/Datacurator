"""
Main entry point for the Data Curator application.
"""
import os
import sys
import click
from pathlib import Path
import json
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import get_settings, ensure_directories
from src.core.database import init_db
from src.core.logging import logger
from src.ingestion.web_scraper import WebScraper
from src.ingestion.advanced_web_scraper import AdvancedWebScraper
from src.ingestion.file_processor import process_file, process_directory
from src.models.api import WebScrapingConfig


class DataCurator:
    """Main Data Curator application class."""
    
    def __init__(self):
        """Initialize the Data Curator application."""
        # Ensure required directories exist
        ensure_directories()
        
        # Initialize settings
        self.settings = get_settings()
        
        # Initialize database
        init_db()
        
        logger.info(f"Data Curator {self.settings.APP_VERSION} initialized")

    async def scrape_websites(self, urls: List[str], selectors: Dict[str, str] = None, 
                             follow_links: bool = False, max_pages: int = 10,
                             use_advanced: bool = True) -> List[Dict[str, Any]]:
        """
        Scrape websites and return the collected data.
        
        Args:
            urls: List of URLs to scrape
            selectors: CSS selectors for extracting data
            follow_links: Whether to follow links on the pages
            max_pages: Maximum number of pages to scrape
            use_advanced: Whether to use the advanced scraper
            
        Returns:
            List of scraped data items
        """
        if use_advanced:
            async with AdvancedWebScraper() as scraper:
                if len(urls) == 1 and follow_links:
                    # Use crawling for single URL with following links
                    return await scraper.crawl_website(
                        start_url=urls[0],
                        max_pages=max_pages,
                        selectors=selectors,
                        use_playwright=False
                    )
                else:
                    # Use regular scraping for multiple URLs
                    url_dicts = [{'url': url} for url in urls]
                    return scraper.scrape_urls_batch(url_dicts, None, "Scraping URLs")
        else:
            # Use legacy scraper
            async with WebScraper() as scraper:
                return await scraper.scrape_urls(
                    urls=urls,
                    selectors=selectors or {},
                    use_playwright=False
                )
    
    def scrape_from_file(self, url_file: str, output_path: str, max_sub_urls: int = 100) -> None:
        """
        Run the advanced scraping pipeline from a URL file.
        
        Args:
            url_file: Path to JSONL file with URLs
            output_path: Path to save results
            max_sub_urls: Maximum sub-URLs per main URL
        """
        scraper = AdvancedWebScraper()
        try:
            scraper.run_pipeline(url_file, output_path, max_sub_urls)
        finally:
            scraper.cleanup_drivers()
    
    def process_files(self, file_paths: List[Union[str, Path]], 
                     recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Process files and return the collected data.
        
        Args:
            file_paths: List of file paths to process
            recursive: Whether to process directories recursively
            
        Returns:
            List of processed data items
        """
        results = []
        
        for path in file_paths:
            path = Path(path)
            
            if path.is_dir():
                # Process all files in directory
                for result in process_directory(path, recursive=recursive):
                    results.append(result)
            else:
                # Process single file
                result = process_file(path)
                results.append(result)
        
        return results


# CLI commands
@click.group()
def cli():
    """Data Curator CLI for processing and cleaning data for LLM training."""
    pass


@cli.command("scrape")
@click.argument("urls", nargs=-1, required=True)
@click.option("--selector", "-s", multiple=True, help="CSS selector in format field:selector")
@click.option("--follow-links/--no-follow-links", default=False, help="Follow links on the pages")
@click.option("--max-pages", type=int, default=10, help="Maximum number of pages to scrape")
@click.option("--output", "-o", type=click.Path(), help="Output file path (JSON)")
@click.option("--advanced/--basic", default=True, help="Use advanced scraper with fallbacks")
def scrape_command(urls, selector, follow_links, max_pages, output, advanced):
    """Scrape websites and collect data."""
    curator = DataCurator()
    
    # Parse selectors
    selectors = {}
    for s in selector:
        parts = s.split(":", 1)
        if len(parts) == 2:
            selectors[parts[0]] = parts[1]
    
    click.echo(f"Scraping {len(urls)} URLs with {'advanced' if advanced else 'basic'} scraper...")
    
    # Run async scraping
    data = asyncio.run(curator.scrape_websites(
        urls=urls,
        selectors=selectors if selectors else None,
        follow_links=follow_links,
        max_pages=max_pages,
        use_advanced=advanced
    ))
    
    click.echo(f"Collected {len(data)} data items")
    
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        click.echo(f"Data saved to {output}")
    else:
        for item in data[:5]:
            if isinstance(item, dict):
                url = item.get('url', 'N/A')
                content_len = len(str(item.get('content', item.get('text', ''))))
                click.echo(f"{url} - {content_len} chars")
        if len(data) > 5:
            click.echo(f"... and {len(data) - 5} more items")


@cli.command("scrape-file")
@click.argument("url_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output JSONL file path")
@click.option("--max-sub-urls", type=int, default=100, help="Maximum sub-URLs per main URL")
def scrape_file_command(url_file, output, max_sub_urls):
    """Scrape URLs from a JSONL file using the advanced pipeline."""
    curator = DataCurator()
    
    click.echo(f"Running advanced scraping pipeline from {url_file}...")
    click.echo(f"Output will be saved to: {output}")
    click.echo(f"Max sub-URLs per main URL: {max_sub_urls}")
    
    try:
        curator.scrape_from_file(url_file, output, max_sub_urls)
        click.echo("✅ Scraping pipeline completed successfully!")
    except Exception as e:
        click.echo(f"❌ Error during scraping: {e}", err=True)
        sys.exit(1)


@cli.command("clean-advanced")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output JSONL file path")
def clean_advanced_command(input_file, output):
    """Clean text using advanced methods with grammar correction and entity filtering."""
    from src.core.advanced_cleaner import AdvancedTextCleaner
    
    click.echo(f"Running advanced text cleaning on {input_file}...")
    click.echo(f"Output will be saved to: {output}")
    
    try:
        with AdvancedTextCleaner() as cleaner:
            cleaned_count = cleaner.run_cleaning_pipeline(input_file, output)
            
        click.echo(f"✅ Advanced cleaning completed: {cleaned_count} texts processed")
        
    except Exception as e:
        click.echo(f"❌ Error during advanced cleaning: {e}", err=True)
        sys.exit(1)


@cli.command("clean-texts")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output JSONL file path")
@click.option("--advanced/--basic", default=False, help="Use advanced cleaning with grammar correction")
@click.option("--min-length", type=int, default=0, help="Minimum content length")
@click.option("--max-length", type=int, default=0, help="Maximum content length (0 for no limit)")
@click.option("--remove-html/--keep-html", default=True, help="Remove HTML tags")
def clean_texts_command(input_file, output, advanced, min_length, max_length, remove_html):
    """Clean texts from a JSONL file with various options."""
    from src.core.cleaner import TextCleaner
    from src.core.advanced_cleaner import AdvancedTextCleaner
    
    click.echo(f"Cleaning texts from {input_file} using {'advanced' if advanced else 'basic'} cleaning...")
    click.echo(f"Output will be saved to: {output}")
    
    try:
        if advanced:
            # Use advanced cleaner
            with AdvancedTextCleaner() as cleaner:
                texts = cleaner.load_texts_from_jsonl(input_file)
                cleaned_texts = cleaner.clean_texts_batch(texts)
                cleaner.save_texts_to_jsonl(cleaned_texts, output)
                
            click.echo(f"✅ Advanced cleaning completed: {len(cleaned_texts)} out of {len(texts)} texts cleaned")
        else:
            # Use basic cleaner
            import json
            
            cleaner = TextCleaner(
                remove_html=remove_html,
                min_length=min_length if min_length > 0 else None,
                max_length=max_length if max_length > 0 else None,
            )
            
            texts = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if 'text' in entry:
                            texts.append(entry['text'])
                    except json.JSONDecodeError:
                        continue
            
            cleaned_texts = []
            for text in texts:
                cleaned = cleaner.clean_text(text)
                if cleaned:
                    cleaned_texts.append(cleaned)
            
            # Save cleaned texts
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                for text in cleaned_texts:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            
            click.echo(f"✅ Basic cleaning completed: {len(cleaned_texts)} out of {len(texts)} texts cleaned")
        
    except Exception as e:
        click.echo(f"❌ Error during cleaning: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error during scraping: {e}", err=True)
        sys.exit(1)
@click.option("--follow-links/--no-follow-links", default=False, help="Follow links on the pages")
@click.option("--max-pages", type=int, default=10, help="Maximum number of pages to scrape")
@click.option("--output", "-o", type=click.Path(), help="Output file path (JSON)")
def scrape_command(urls, selector, follow_links, max_pages, output):
    """Scrape websites and collect data."""
    curator = DataCurator()
    
    # Parse selectors
    selectors = {}
    for s in selector:
        parts = s.split(":", 1)
        if len(parts) == 2:
            selectors[parts[0]] = parts[1]
    
    if not selectors:
        click.echo("Error: At least one selector is required", err=True)
        return
    
    click.echo(f"Scraping {len(urls)} URLs with {len(selectors)} selectors...")
    
    data = curator.scrape_websites(
        urls=urls,
        selectors=selectors,
        follow_links=follow_links,
        max_pages=max_pages,
    )
    
    click.echo(f"Collected {len(data)} data items")
    
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        click.echo(f"Data saved to {output}")
    else:
        click.echo(json.dumps(data[:5], indent=2, ensure_ascii=False))
        if len(data) > 5:
            click.echo(f"... and {len(data) - 5} more items")


@cli.command("process")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", default=False, help="Process directories recursively")
@click.option("--output", "-o", type=click.Path(), help="Output file path (JSON)")
def process_command(paths, recursive, output):
    """Process files and collect data."""
    curator = DataCurator()
    
    click.echo(f"Processing {len(paths)} files/directories...")
    
    data = curator.process_files(
        file_paths=paths,
        recursive=recursive,
    )
    
    click.echo(f"Processed {len(data)} files")
    
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        click.echo(f"Data saved to {output}")
    else:
        for item in data[:5]:
            click.echo(f"{item['file_path']} - {item['mime_type']} ({item['file_size']} bytes)")
        if len(data) > 5:
            click.echo(f"... and {len(data) - 5} more files")


@cli.command("clean")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file path")
@click.option("--remove-html/--keep-html", default=True, help="Remove HTML tags")
@click.option("--min-length", type=int, default=0, help="Minimum content length")
@click.option("--max-length", type=int, default=0, help="Maximum content length (0 for no limit)")
@click.option("--deduplicate/--no-deduplicate", default=True, help="Remove duplicate content")
def clean_command(input_file, output, remove_html, min_length, max_length, deduplicate):
    """Clean and process data for LLM training."""
    click.echo(f"Cleaning data from {input_file}...")
    click.echo("This feature is not yet implemented")


@cli.command("init-db")
def init_db_command():
    """Initialize the database."""
    curator = DataCurator()
    click.echo("Database initialized")


if __name__ == "__main__":
    cli()

"""
Main entry point for the Data Curator application.
"""
import os
import sys
import click
from pathlib import Path
import json
import datetime
from typing import Dict, Any, List, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import get_settings, ensure_directories
from src.core.database import init_db
from src.core.logging import logger
from src.ingestion.web_scraper import WebScraper, scrape_website
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

    def scrape_websites(self, urls: List[str], selectors: Dict[str, str], 
                       follow_links: bool = False, max_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape websites and return the collected data.
        
        Args:
            urls: List of URLs to scrape
            selectors: CSS selectors for extracting data
            follow_links: Whether to follow links on the pages
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of scraped data items
        """
        config = WebScrapingConfig(
            urls=urls,
            selectors=selectors,
            follow_links=follow_links,
            max_pages=max_pages,
        )
        
        return scrape_website(config)
    
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

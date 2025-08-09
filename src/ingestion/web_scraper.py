"""
Web scraping utilities for the data curation system.
"""
import time
i    def _can_fetch(self, url: Union[str, HttpUrl]) -> bool:
        """
        Check if we're allowed to fetch a URL according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            Whether we're allowed to fetch the URL
        """
        if not self.respect_robots:
            return True
        
        try:
            url_str = str(url)
            parsed_url = urlparse(url_str)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"rom typing import List, Dict, Any, Optional, Union, Generator
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import random
from datetime import datetime
from playwright.sync_api import sync_playwright, Page, Browser
from loguru import logger

from ..core.config import get_settings
from ..models.api import WebScrapingConfig

settings = get_settings()


class WebScraper:
    """Web scraper for collecting data from websites."""
    
    def __init__(
        self,
        config: WebScrapingConfig,
        use_playwright: bool = False,
        respect_robots: bool = True,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize the web scraper.
        
        Args:
            config: Scraping configuration
            use_playwright: Whether to use Playwright (for JS-heavy sites)
            respect_robots: Whether to respect robots.txt
            user_agent: Custom user agent
        """
        self.config = config
        self.use_playwright = use_playwright
        self.respect_robots = respect_robots
        self.user_agent = user_agent or settings.DEFAULT_REQUEST_HEADERS["User-Agent"]
        self.headers = {**settings.DEFAULT_REQUEST_HEADERS, "User-Agent": self.user_agent}
        
        # Initialize state
        self.visited_urls = set()
        self.collected_data = []
        
        # Playwright browser (only initialized if needed)
        self.browser = None
        self.page = None
    
    def __enter__(self):
        """Context manager entry."""
        if self.use_playwright:
            self._setup_playwright()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.browser:
            self.browser.close()
    
    def _setup_playwright(self):
        """Set up Playwright browser."""
        playwright = sync_playwright().start()
        self.browser = playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page(
            user_agent=self.user_agent,
            viewport={"width": 1280, "height": 800}
        )
    
    def _can_fetch(self, url: str) -> bool:
        """
        Check if we're allowed to fetch the URL according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            Whether we're allowed to fetch the URL
        """
        if not self.respect_robots:
            return True
        
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            response = requests.get(robots_url, headers=self.headers, timeout=5)
            if response.status_code != 200:
                return True
            
            # Simple robots.txt parsing
            lines = response.text.split('\n')
            user_agent_match = False
            
            for line in lines:
                line = line.strip().lower()
                
                if line.startswith('user-agent:'):
                    agent = line[11:].strip()
                    if agent == '*' or self.user_agent.lower() in agent:
                        user_agent_match = True
                    else:
                        user_agent_match = False
                
                if user_agent_match and line.startswith('disallow:'):
                    path = line[9:].strip()
                    if path and parsed_url.path.startswith(path):
                        return False
            
            return True
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True
    
    def _fetch_with_requests(self, url: str) -> Optional[str]:
        """
        Fetch URL content using requests.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=settings.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url} with requests: {e}")
            return None
    
    def _fetch_with_playwright(self, url: str) -> Optional[str]:
        """
        Fetch URL content using Playwright.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        if not self.page:
            self._setup_playwright()
        
        try:
            self.page.goto(url, wait_until="networkidle", timeout=30000)
            # Allow dynamic content to load
            self.page.wait_for_timeout(2000)
            return self.page.content()
        except Exception as e:
            logger.error(f"Error fetching {url} with Playwright: {e}")
            return None
    
    def _extract_data(self, html: str, url: str) -> List[Dict[str, Any]]:
        """
        Extract data from HTML using provided selectors.
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            List of extracted data items
        """
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Handle different selector types:
        # 1. Single item extraction (use all selectors on the page)
        # 2. Multiple items extraction (use item_selector to find items, then apply other selectors)
        
        if 'item_selector' in self.config.selectors:
            # Multiple items extraction
            item_selector = self.config.selectors.pop('item_selector')
            items = soup.select(item_selector)
            
            for item in items:
                data = {'source_url': url, 'scraped_at': datetime.utcnow().isoformat()}
                
                for field, selector in self.config.selectors.items():
                    elements = item.select(selector)
                    if elements:
                        data[field] = elements[0].get_text(strip=True)
                    else:
                        data[field] = None
                
                results.append(data)
            
            # Restore the config
            self.config.selectors['item_selector'] = item_selector
        else:
            # Single item extraction
            data = {'source_url': url, 'scraped_at': datetime.utcnow().isoformat()}
            
            for field, selector in self.config.selectors.items():
                elements = soup.select(selector)
                if elements:
                    data[field] = elements[0].get_text(strip=True)
                else:
                    data[field] = None
            
            results.append(data)
        
        return results
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """
        Extract links from HTML for crawling.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of extracted links
        """
        if not html or not self.config.follow_links:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
            
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            
            # Only include links from the same domain
            if parsed_url.netloc == base_domain:
                links.append(absolute_url)
        
        return links
    
    def scrape_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            List of extracted data items
        """
        if url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        
        if not self._can_fetch(url):
            logger.info(f"Skipping {url} (disallowed by robots.txt)")
            return []
        
        logger.info(f"Scraping {url}")
        
        # Fetch content
        if self.use_playwright:
            html = self._fetch_with_playwright(url)
        else:
            html = self._fetch_with_requests(url)
        
        if not html:
            return []
        
        # Extract data
        data = self._extract_data(html, url)
        
        # Add delay to be polite
        if self.config.delay:
            delay = self.config.delay
            # Add some randomization to avoid detection
            delay += random.uniform(-0.5, 0.5) * delay * 0.2
            delay = max(0.1, delay)  # Ensure minimum delay
            time.sleep(delay)
        
        return data
    
    def scrape(self) -> Generator[Dict[str, Any], None, None]:
        """
        Scrape all URLs in the configuration.
        
        Yields:
            Extracted data items
        """
        to_visit = list(self.config.urls)
        pages_scraped = 0
        
        while to_visit and pages_scraped < self.config.max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            data_items = self.scrape_url(url)
            for item in data_items:
                yield item
                self.collected_data.append(item)
            
            # Extract and add new links to visit
            if self.config.follow_links:
                if self.use_playwright:
                    html = self.page.content() if self.page else None
                else:
                    html = self._fetch_with_requests(url)
                
                if html:
                    new_links = self._extract_links(html, url)
                    for link in new_links:
                        if link not in self.visited_urls and link not in to_visit:
                            to_visit.append(link)
            
            pages_scraped += 1


def scrape_website(config: WebScrapingConfig) -> List[Dict[str, Any]]:
    """
    Scrape a website using the provided configuration.
    
    Args:
        config: Scraping configuration
        
    Returns:
        List of extracted data items
    """
    use_playwright = any('javascript' in str(url) for url in config.urls)
    
    with WebScraper(config, use_playwright=use_playwright) as scraper:
        data = list(scraper.scrape())
    
    # Convert data for JSON serialization
    json_ready_data = []
    for item in data:
        json_item = {}
        for key, value in item.items():
            if hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool, type(None))):
                json_item[key] = str(value)
            else:
                json_item[key] = value
        json_ready_data.append(json_item)
    
    return json_ready_data

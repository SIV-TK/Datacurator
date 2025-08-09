"""
Web scraping utilities for the data curation system.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import aiohttp
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class WebScraper:
    """Web scraper for collecting data from websites with basic functionality."""
    
    def __init__(self):
        self.config = get_settings()
        self.scraped_urls: Set[str] = set()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        
        return text.strip()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and should be scraped."""
        if not url:
            return False
            
        parsed = urlparse(url)
        if not parsed.netloc:
            return False
            
        # Skip certain file types
        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        return True
    
    async def _extract_with_playwright(self, url: str, selectors: Dict[str, str]) -> Dict[str, any]:
        """Extract content using Playwright for JavaScript-heavy sites."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(url, wait_until="networkidle")
                
                result = {
                    'url': url,
                    'title': await page.title(),
                    'content': {}
                }
                
                # Extract content using provided selectors
                for name, selector in selectors.items():
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            content = await element.inner_text()
                            result['content'][name] = self._clean_text(content)
                    except Exception as e:
                        logger.warning(f"Failed to extract '{name}' from {url}: {e}")
                        result['content'][name] = ""
                
                # If no selectors provided, get main content
                if not selectors:
                    body = await page.query_selector('body')
                    if body:
                        result['content']['text'] = self._clean_text(await body.inner_text())
                
                # Extract links for crawling
                links = await page.query_selector_all('a[href]')
                result['links'] = []
                for link in links:
                    href = await link.get_attribute('href')
                    if href:
                        full_url = urljoin(url, href)
                        if self._is_valid_url(full_url):
                            result['links'].append(full_url)
                
                return result
                
            except Exception as e:
                logger.error(f"Playwright extraction failed for {url}: {e}")
                return {'url': url, 'content': {}, 'links': [], 'error': str(e)}
            finally:
                await browser.close()
    
    async def _extract_with_aiohttp(self, url: str, selectors: Dict[str, str]) -> Dict[str, any]:
        """Extract content using aiohttp and BeautifulSoup for simple sites."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {'url': url, 'content': {}, 'links': [], 'error': f'HTTP {response.status}'}
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                result = {
                    'url': url,
                    'title': soup.title.string if soup.title else "",
                    'content': {}
                }
                
                # Extract content using provided selectors
                for name, selector in selectors.items():
                    try:
                        elements = soup.select(selector)
                        content = " ".join([elem.get_text() for elem in elements])
                        result['content'][name] = self._clean_text(content)
                    except Exception as e:
                        logger.warning(f"Failed to extract '{name}' from {url}: {e}")
                        result['content'][name] = ""
                
                # If no selectors provided, get main content
                if not selectors:
                    # Try to find main content areas
                    main_selectors = ['main', 'article', '.content', '#content', 'body']
                    for sel in main_selectors:
                        elements = soup.select(sel)
                        if elements:
                            result['content']['text'] = self._clean_text(elements[0].get_text())
                            break
                
                # Extract links
                result['links'] = []
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(url, link['href'])
                    if self._is_valid_url(full_url):
                        result['links'].append(full_url)
                
                return result
                
        except Exception as e:
            logger.error(f"aiohttp extraction failed for {url}: {e}")
            return {'url': url, 'content': {}, 'links': [], 'error': str(e)}
    
    async def scrape_url(self, url: str, selectors: Dict[str, str] = None, 
                        use_playwright: bool = False) -> Dict[str, any]:
        """
        Scrape a single URL.
        
        Args:
            url: The URL to scrape
            selectors: Dictionary of name -> CSS selector mappings
            use_playwright: Whether to use Playwright (for JS-heavy sites)
            
        Returns:
            Dictionary with scraped content
        """
        if not self._is_valid_url(url):
            return {'url': url, 'content': {}, 'links': [], 'error': 'Invalid URL'}
        
        if url in self.scraped_urls:
            return {'url': url, 'content': {}, 'links': [], 'error': 'Already scraped'}
        
        selectors = selectors or {}
        
        if use_playwright:
            result = await self._extract_with_playwright(url, selectors)
        else:
            result = await self._extract_with_aiohttp(url, selectors)
        
        self.scraped_urls.add(url)
        return result
    
    async def scrape_urls(self, urls: List[str], selectors: Dict[str, str] = None,
                         use_playwright: bool = False, max_concurrent: int = 10) -> List[Dict[str, any]]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            selectors: Dictionary of name -> CSS selector mappings
            use_playwright: Whether to use Playwright
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of dictionaries with scraped content
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_url(url, selectors, use_playwright)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scraping failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def crawl_website(self, start_url: str, max_pages: int = 100, 
                          selectors: Dict[str, str] = None, use_playwright: bool = False) -> List[Dict[str, any]]:
        """
        Crawl a website starting from a URL.
        
        Args:
            start_url: Starting URL
            max_pages: Maximum number of pages to crawl
            selectors: Dictionary of name -> CSS selector mappings
            use_playwright: Whether to use Playwright
            
        Returns:
            List of dictionaries with scraped content
        """
        to_visit = [start_url]
        visited = set()
        results = []
        
        base_domain = urlparse(start_url).netloc
        
        while to_visit and len(results) < max_pages:
            current_batch = to_visit[:10]  # Process in batches
            to_visit = to_visit[10:]
            
            batch_results = await self.scrape_urls(current_batch, selectors, use_playwright)
            
            for result in batch_results:
                if 'error' not in result:
                    results.append(result)
                    visited.add(result['url'])
                    
                    # Add new links from same domain
                    for link in result.get('links', []):
                        if (urlparse(link).netloc == base_domain and 
                            link not in visited and 
                            link not in to_visit):
                            to_visit.append(link)
        
        return results

"""
Advanced web scraping utilities with multiple extraction methods and intelligent fallbacks.
"""
import os
import json
import html
import re
import logging
import validators
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from newspaper import Article, Config
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from langdetect import detect
from trafilatura import extract
from contextlib import contextmanager
import time
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Set, Tuple, Any
import asyncio
import aiohttp
from playwright.async_api import async_playwright
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class AdvancedWebScraper:
    """Advanced web scraper with multiple extraction methods and intelligent fallbacks."""
    
    def __init__(self):
        self.config = get_settings()
        self.scraped_urls: Set[str] = set()
        self.session = None
        self._driver_pool = []
        
        # Configuration with defaults
        self.num_threads = self.config.SCRAPER_NUM_THREADS
        self.use_selenium_fallback = self.config.SCRAPER_USE_SELENIUM_FALLBACK
        self.rate_limit_delay = self.config.SCRAPER_RATE_LIMIT_DELAY
        self.target_language = self.config.SCRAPER_TARGET_LANGUAGE
        self.max_sub_urls = self.config.SCRAPER_MAX_SUB_URLS
        
        # Domains and extensions to exclude
        self.exclude_domains = {
            'imgur.com', 'instagram.com', 'discord.gg', 'youtube.com', 'twitter.com',
            'reddit.com', 'spotify.com', 'github.com', 'facebook.com', 'wikipedia.org',
            'giphy.com', 'vimeo.com', 'twitch.tv', 'soundcloud.com', 'google.com'
        }
        
        self.exclude_extensions = (
            '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.mp4', '.mp3', '.zip', '.rar',
            '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.exe', '.bin'
        )
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.cleanup_drivers()
    
    def init_selenium_driver(self):
        """Initialize a Selenium WebDriver with optimal settings."""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        return webdriver.Chrome(options=options)
    
    @contextmanager
    def get_selenium_driver(self):
        """Context manager for Selenium driver with pooling."""
        if not self._driver_pool:
            self._driver_pool.append(self.init_selenium_driver())
        driver = self._driver_pool[0]
        try:
            yield driver
        finally:
            pass  # Reuse driver
    
    def cleanup_drivers(self):
        """Clean up all Selenium drivers."""
        for driver in self._driver_pool:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
        self._driver_pool.clear()
    
    def should_exclude(self, url: str) -> bool:
        """Check if URL should be excluded from scraping."""
        if not url or len(url) <= 8 or ' ' in url or not validators.url(url):
            return True

        parsed = urlparse(url)
        domain = f"{parsed.netloc}".lower()
        if domain in self.exclude_domains or any(domain.endswith(d) for d in self.exclude_domains):
            return True

        if any(url.lower().split('?')[0].endswith(ext) for ext in self.exclude_extensions):
            return True

        return False
    
    def extract_sub_urls(self, url: str, html_content: str, max_sub_urls: int = None) -> List[str]:
        """Extract sub-URLs from HTML content."""
        if max_sub_urls is None:
            max_sub_urls = self.max_sub_urls
            
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            sub_urls = set()
            base_domain = urlparse(url).netloc
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == base_domain and not self.should_exclude(full_url):
                    sub_urls.add(full_url)
                    
            return list(sub_urls)[:max_sub_urls]
        except Exception as e:
            logger.error(f"Failed to extract sub-URLs from {url}: {e}")
            return []
    
    def clean_text(self, text: str) -> Optional[str]:
        """Clean and validate extracted text."""
        if not text:
            return None
            
        # Decode HTML entities
        text = html.unescape(text)
        text = text.replace('\xa0', ' ')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text)
        text = text.strip()

        # Quality filters
        if len(text.split()) < 100:
            return None
        if sum(1 for c in text if not c.isalnum() and c not in " .,!?'\"") / len(text) > 0.1:
            return None
        if text.count("http") > 3 or any(sym in text for sym in ['{', '}', '=', 'function']):
            return None

        return text
    
    def scrape_with_newspaper(self, url: str) -> Tuple[Optional[str], str, List[str]]:
        """Scrape using newspaper3k library."""
        try:
            config = Config()
            config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            text = article.text.strip()
            html_content = article.html
            sub_urls = self.extract_sub_urls(url, html_content)
            
            return text, html_content, sub_urls
        except Exception as e:
            logger.warning(f"Newspaper failed for {url}: {e}")
            return None, "", []
    
    def scrape_with_selenium(self, url: str) -> Tuple[Optional[str], str, List[str]]:
        """Scrape using Selenium WebDriver."""
        try:
            with self.get_selenium_driver() as driver:
                driver.get(url)
                time.sleep(2)  # Wait for page load
                
                # Try trafilatura first for better text extraction
                html_content = driver.page_source
                text = extract(html_content)
                
                # Fallback to body text if trafilatura fails
                if not text or len(text) < 100:
                    body_element = driver.find_element(By.TAG_NAME, 'body')
                    text = body_element.text
                
                sub_urls = self.extract_sub_urls(url, html_content)
                return text, html_content, sub_urls
        except Exception as e:
            logger.error(f"Selenium failed for {url}: {e}")
            return None, "", []
    
    async def scrape_with_playwright(self, url: str, selectors: Dict[str, str] = None) -> Dict[str, Any]:
        """Scrape using Playwright for JavaScript-heavy sites."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                result = {
                    'url': url,
                    'title': await page.title(),
                    'content': {}
                }
                
                # Extract content using provided selectors
                if selectors:
                    for name, selector in selectors.items():
                        try:
                            element = await page.query_selector(selector)
                            if element:
                                content = await element.inner_text()
                                result['content'][name] = self.clean_text(content)
                        except Exception as e:
                            logger.warning(f"Failed to extract '{name}' from {url}: {e}")
                            result['content'][name] = ""
                else:
                    # Default extraction
                    body = await page.query_selector('body')
                    if body:
                        text = await body.inner_text()
                        result['content']['text'] = self.clean_text(text)
                
                # Extract links
                links = await page.query_selector_all('a[href]')
                result['links'] = []
                for link in links:
                    href = await link.get_attribute('href')
                    if href:
                        full_url = urljoin(url, href)
                        if not self.should_exclude(full_url):
                            result['links'].append(full_url)
                
                return result
                
            except Exception as e:
                logger.error(f"Playwright extraction failed for {url}: {e}")
                return {'url': url, 'content': {}, 'links': [], 'error': str(e)}
            finally:
                await browser.close()
    
    def scrape_article(self, url: str, output_path: str = None) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Scrape a single article with multiple fallback methods.
        
        Returns:
            Tuple of (article_data, sub_urls)
        """
        if self.should_exclude(url):
            return None, []
        
        text = None
        sub_urls = []
        
        # Method 1: Try newspaper3k first
        text, html_content, sub_urls = self.scrape_with_newspaper(url)
        
        # Method 2: Selenium fallback if newspaper fails
        if not text and self.use_selenium_fallback:
            text, html_content, sub_urls = self.scrape_with_selenium(url)
        
        # Process and validate text
        if not text:
            return None, sub_urls
        
        try:
            # Language detection
            if detect(text) != self.target_language:
                logger.info(f"Language mismatch for {url}")
                return None, sub_urls
        except:
            # Language detection failed, proceed anyway
            pass

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return None, sub_urls

        article_data = {
            "url": url,
            "text": cleaned_text,
            "word_count": len(cleaned_text.split()),
            "extracted_at": time.time()
        }

        # Save immediately to avoid memory buildup
        if output_path:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(article_data, ensure_ascii=False) + '\n')

        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        return article_data, sub_urls
    
    def scrape_urls_batch(self, urls: List[Dict[str, str]], output_path: str, description: str = "Scraping") -> List[Dict[str, Any]]:
        """Scrape multiple URLs using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self.scrape_article, url_dict['url'], output_path): url_dict['url']
                for url_dict in urls
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=description):
                try:
                    result, sub_urls = future.result()
                    if result:
                        results.append(result)
                        # Add sub-URLs for potential future processing
                        results.extend([{"url": sub_url} for sub_url in sub_urls])
                except Exception as e:
                    url = futures[future]
                    logger.error(f"Future failed for {url}: {e}")
                    continue
        
        return results
    
    def load_processed_urls(self, output_path: str) -> Set[str]:
        """Load already processed URLs from output file."""
        processed = set()
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        processed.add(doc.get('url', ''))
                    except json.JSONDecodeError:
                        continue
        return processed
    
    def load_urls_from_file(self, jsonl_path: str) -> List[Dict[str, str]]:
        """Load URLs from JSONL file."""
        urls = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'url' in entry and not self.should_exclude(entry['url']):
                        urls.append(entry)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON line: {line.strip()}")
                    continue
        logger.info(f"Loaded {len(urls)} valid URLs.")
        return urls
    
    async def crawl_website(self, start_url: str, max_pages: int = 100, 
                          selectors: Dict[str, str] = None, use_playwright: bool = False) -> List[Dict[str, Any]]:
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
            
            if use_playwright:
                for url in current_batch:
                    if url not in visited:
                        result = await self.scrape_with_playwright(url, selectors)
                        if 'error' not in result:
                            results.append(result)
                            visited.add(url)
                            
                            # Add new links from same domain
                            for link in result.get('links', []):
                                if (urlparse(link).netloc == base_domain and 
                                    link not in visited and 
                                    link not in to_visit):
                                    to_visit.append(link)
            else:
                # Use the traditional scraping approach
                url_dicts = [{'url': url} for url in current_batch if url not in visited]
                batch_results = self.scrape_urls_batch(url_dicts, None, "Crawling")
                
                for result in batch_results:
                    if result and 'url' in result:
                        results.append(result)
                        visited.add(result['url'])
        
        return results
    
    def run_pipeline(self, url_file: str, output_path: str, max_sub_urls: int = None):
        """
        Run the complete scraping pipeline.
        
        Args:
            url_file: Path to JSONL file with URLs
            output_path: Path to save results
            max_sub_urls: Maximum sub-URLs per main URL
        """
        try:
            if max_sub_urls is None:
                max_sub_urls = self.max_sub_urls
                
            # Load initial URLs
            main_urls = self.load_urls_from_file(url_file)
            processed_urls = self.load_processed_urls(output_path)
            main_urls = [url for url in main_urls if url['url'] not in processed_urls]
            
            # Step 1: Collect sub-URLs
            sub_urls = []
            logger.info("Collecting sub-URLs...")
            for url in tqdm(main_urls, desc="Collecting sub-URLs"):
                try:
                    text, html_content, extracted_sub_urls = self.scrape_with_newspaper(url['url'])
                    sub_urls.extend([{"url": sub_url} for sub_url in extracted_sub_urls])
                except Exception as e:
                    logger.warning(f"Failed to collect sub-URLs from {url['url']}: {e}")
                    if self.use_selenium_fallback:
                        try:
                            text, html_content, extracted_sub_urls = self.scrape_with_selenium(url['url'])
                            sub_urls.extend([{"url": sub_url} for sub_url in extracted_sub_urls])
                        except Exception as e:
                            logger.error(f"Selenium failed for sub-URL collection from {url['url']}: {e}")
                            continue
                time.sleep(self.rate_limit_delay)
            
            sub_urls = [url for url in sub_urls if url['url'] not in processed_urls]
            logger.info(f"Collected {len(sub_urls)} sub-URLs.")
            
            # Step 2: Scrape sub-URLs
            self.scrape_urls_batch(sub_urls, output_path, "Scraping sub-URLs")
            
            # Step 3: Scrape main URLs
            self.scrape_urls_batch(main_urls, output_path, "Scraping main URLs")
            
            logger.info(f"Results saved to: {output_path}")
        finally:
            self.cleanup_drivers()


# Legacy class for backward compatibility
class WebScraper(AdvancedWebScraper):
    """Legacy WebScraper class - now uses AdvancedWebScraper."""
    
    def __init__(self):
        super().__init__()
        logger.warning("WebScraper is deprecated. Use AdvancedWebScraper instead.")
    
    async def scrape_url(self, url: str, selectors: Dict[str, str] = None, 
                        use_playwright: bool = False) -> Dict[str, Any]:
        """Legacy method - redirects to new implementation."""
        if use_playwright and selectors:
            return await self.scrape_with_playwright(url, selectors)
        else:
            result, _ = self.scrape_article(url)
            return result or {'url': url, 'content': {}, 'error': 'Failed to scrape'}
    
    async def scrape_urls(self, urls: List[str], selectors: Dict[str, str] = None,
                         use_playwright: bool = False, max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """Legacy method - redirects to new implementation."""
        if use_playwright:
            results = []
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def scrape_with_semaphore(url):
                async with semaphore:
                    return await self.scrape_with_playwright(url, selectors)
            
            tasks = [scrape_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return [r for r in results if not isinstance(r, Exception)]
        else:
            url_dicts = [{'url': url} for url in urls]
            return self.scrape_urls_batch(url_dicts, None, "Scraping URLs")

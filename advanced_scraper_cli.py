#!/usr/bin/env python3
"""
Standalone script to run the advanced web scraping pipeline.
This script includes the configuration loading and pipeline execution in one file.
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


# ----------------- CONFIG -----------------
def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    default_config = {
        "num_threads": 8,
        "use_selenium_fallback": True,
        "working_dir": "data",
        "rate_limit_delay": 0.5,
        "target_language": "en"
    }
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            default_config.update(config)
    return default_config


CONFIG = load_config()
NUM_THREADS = CONFIG["num_threads"]
USE_SELENIUM_FALLBACK = CONFIG["use_selenium_fallback"]
WORKING_DIR = CONFIG["working_dir"]
RATE_LIMIT_DELAY = CONFIG["rate_limit_delay"]
TARGET_LANGUAGE = CONFIG["target_language"]

# ------------------------------------------

# Logging configuration
os.makedirs(WORKING_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(WORKING_DIR, 'scraper.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
filelock_logger = logging.getLogger("filelock")
filelock_logger.setLevel(logging.WARNING)

# Domains and extensions to exclude
exclude_domains = {
    'imgur.com', 'instagram.com', 'discord.gg', 'youtube.com', 'twitter.com',
    'reddit.com', 'spotify.com', 'github.com', 'facebook.com', 'wikipedia.org',
    'giphy.com', 'vimeo.com', 'twitch.tv', 'soundcloud.com', 'google.com'
}

exclude_extensions = (
    '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.mp4', '.mp3', '.zip', '.rar',
    '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.exe', '.bin'
)

# Selenium driver pool
_driver_pool = []


def init_selenium_driver():
    """Initialize Selenium WebDriver."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(options=options)


@contextmanager
def get_selenium_driver():
    """Context manager for Selenium driver with pooling."""
    if not _driver_pool:
        _driver_pool.append(init_selenium_driver())
    driver = _driver_pool[0]
    try:
        yield driver
    finally:
        pass  # Reuse driver


def cleanup_drivers():
    """Clean up all Selenium drivers."""
    for driver in _driver_pool:
        driver.quit()
    _driver_pool.clear()


def should_exclude(url):
    """Check if URL should be excluded from scraping."""
    if not url or len(url) <= 8 or ' ' in url or not validators.url(url):
        return True

    parsed = urlparse(url)
    domain = f"{parsed.netloc}".lower()
    if domain in exclude_domains or any(domain.endswith(d) for d in exclude_domains):
        return True

    if any(url.lower().split('?')[0].endswith(ext) for ext in exclude_extensions):
        return True

    return False


def extract_sub_urls(url, html_content, max_sub_urls):
    """Extract sub-URLs from HTML content."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        sub_urls = set()
        base_domain = urlparse(url).netloc
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(url, href)
            if urlparse(full_url).netloc == base_domain and not should_exclude(full_url):
                sub_urls.add(full_url)
        return list(sub_urls)[:max_sub_urls]
    except Exception as e:
        logging.error(f"Failed to extract sub-URLs from {url}: {e}")
        return []


def prompt_for_url_file():
    """Prompt user for URL file path."""
    return input("📥 Enter path to URL list (.jsonl): ").strip()


def prompt_for_max_sub_urls():
    """Prompt user for maximum sub-URLs."""
    try:
        max_sub_urls = int(input("🔢 Enter maximum number of sub-URLs to scrape per URL (default 100): ") or 100)
        return max(max_sub_urls, 0)
    except ValueError:
        print("⚠️ Invalid input, using default of 100 sub-URLs.")
        return 100


def load_urls(jsonl_path):
    """Load URLs from JSONL file."""
    urls = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'url' in entry and not should_exclude(entry['url']):
                    urls.append(entry)
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON line: {line.strip()}")
                continue
    print(f"✅ Loaded {len(urls)} valid URLs.")
    return urls


def load_processed_urls(output_path):
    """Load already processed URLs."""
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


def clean_text(text):
    """Clean and validate extracted text."""
    text = html.unescape(text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text)
    text = text.strip()

    if len(text.split()) < 100:
        return None
    if sum(1 for c in text if not c.isalnum() and c not in " .,!?'\"") / len(text) > 0.1:
        return None
    if text.count("http") > 3 or any(sym in text for sym in ['{', '}', '=', 'function']):
        return None

    return text


def scrape_article(url, max_sub_urls, output_path):
    """Scrape a single article with multiple fallback methods."""
    text = ""
    sub_urls = []

    try:
        # Configure newspaper with user-agent
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text.strip()
        sub_urls = extract_sub_urls(url, article.html, max_sub_urls)
    except Exception as e:
        logging.warning(f"Newspaper failed for {url}: {e}")
        if USE_SELENIUM_FALLBACK:
            try:
                with get_selenium_driver() as driver:
                    driver.get(url)
                    text = extract(driver.page_source) or driver.find_element(By.TAG_NAME, 'body').text
                    sub_urls = extract_sub_urls(url, driver.page_source, max_sub_urls)
            except Exception as e:
                logging.error(f"Selenium failed for {url}: {e}")
                return None, []

    try:
        if not text or len(text) < 100:
            return None, sub_urls
        if detect(text) != TARGET_LANGUAGE:
            return None, sub_urls

        cleaned = clean_text(text)
        if not cleaned:
            return None, sub_urls

        # Save immediately to avoid memory buildup
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"text": cleaned}, ensure_ascii=False) + '\n')

        # Add rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        return {"url": url}, sub_urls
    except Exception as e:
        logging.error(f"Processing failed for {url}: {e}")
        return None, sub_urls


def scrape_urls(urls, max_sub_urls, output_path, desc):
    """Scrape multiple URLs using ThreadPoolExecutor."""
    results = []
    processed_urls = load_processed_urls(output_path)
    urls = [url for url in urls if url['url'] not in processed_urls]

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(scrape_article, url['url'], max_sub_urls, output_path): url['url']
            for url in urls
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result, sub_urls = future.result()
                if result:
                    results.append(result)
                    results.extend([{"url": sub_url} for sub_url in sub_urls])
            except Exception as e:
                logging.error(f"Future failed: {e}")
                continue
    return results


def run_pipeline():
    """Run the complete scraping pipeline."""
    try:
        path = prompt_for_url_file()
        if not os.path.isfile(path):
            print("❌ File not found.")
            return

        max_sub_urls = prompt_for_max_sub_urls()
        os.makedirs(WORKING_DIR, exist_ok=True)
        output_path = os.path.join(WORKING_DIR, "extracted_texts.jsonl")

        # Load initial URLs
        main_urls = load_urls(path)
        processed_urls = load_processed_urls(output_path)
        main_urls = [url for url in main_urls if url['url'] not in processed_urls]

        # Step 1: Scrape sub-URLs
        sub_urls = []
        print("🌐 Collecting sub-URLs...")
        for url in tqdm(main_urls, desc="Collecting sub-URLs"):
            try:
                config = Config()
                config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                article = Article(url['url'], config=config)
                article.download()
                article.parse()
                sub_urls.extend([{"url": sub_url} for sub_url in extract_sub_urls(url['url'], article.html, max_sub_urls)])
            except Exception as e:
                logging.warning(f"Failed to collect sub-URLs from {url['url']}: {e}")
                try:
                    if USE_SELENIUM_FALLBACK:
                        with get_selenium_driver() as driver:
                            driver.get(url['url'])
                            sub_urls.extend([{"url": sub_url} for sub_url in extract_sub_urls(url['url'], driver.page_source, max_sub_urls)])
                except Exception as e:
                    logging.error(f"Selenium failed for sub-URL collection from {url['url']}: {e}")
                    continue
            time.sleep(RATE_LIMIT_DELAY)

        sub_urls = [url for url in sub_urls if url['url'] not in processed_urls]
        print(f"✅ Collected {len(sub_urls)} sub-URLs.")

        # Step 2: Scrape sub-URLs
        scrape_urls(sub_urls, max_sub_urls, output_path, "Scraping sub-URLs")

        # Step 3: Scrape main URLs
        scrape_urls(main_urls, max_sub_urls, output_path, "Scraping main URLs")

        print(f"💾 Results saved to: {output_path}")
    finally:
        cleanup_drivers()


if __name__ == "__main__":
    run_pipeline()

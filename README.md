# kArmas_fram3work4TheLulz
a framework to my Anonymous & Lulzsec friends

# kArmas_fram3work_minimal.py
# Simplified for Termux - Basic Scrapy crawler
# made in l0ve by kArmasec (adapted for Android/Termux)

import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

from scrapy import Spider, Request, signals
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# === COLORS (works in Termux) ===
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def cprint(text: str, color: str = Colors.GREEN):
    print(f"{color}{text}{Colors.RESET}")

# === GLOBALS ===
ua = UserAgent()
BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "kArmasOutPut.txt"

OUTPUT_FILE.write_text(f"""
{Colors.GREEN}kArmas_fram3work MINIMAL - TERMUX EDITION{Colors.RESET}
made in l0ve by kArmasec
Basic Scrapy Crawler (no JS/AI/PDF/DB)
{"="*50}
""")

class SimpleSpider(Spider):
    name = "simple"
    custom_settings = {
        'USER_AGENT': ua.random,
        'DOWNLOAD_DELAY': 1.0,
        'CONCURRENT_REQUESTS': 8,
        'ROBOTSTXT_OBEY': True,
    }

    def __init__(self, start_urls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls or ['https://example.com']

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, callback=self.parse)

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else "No Title"
        text = soup.get_text(separator=' ', strip=True)[:2000]
        links = [urljoin(response.url, a['href']) for a in soup.find_all('a', href=True) if not a['href'].startswith(('mailto:', 'javascript:'))][:50]

        item = {
            'url': response.url,
            'title': title,
            'text_preview': text[:500] + '...',
            'links_found': len(links),
            'crawled_at': datetime.now().isoformat(),
        }

        # Console output
        cprint(f"\n[{Colors.BOLD}{Colors.GREEN}CRAWLED{Colors.RESET}] {response.url[:60]}", Colors.CYAN)
        cprint(f"  Title: {title[:70]}")
        cprint(f"  Links: {len(links)}")

        # File output
        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] CRAWLED: {response.url}\n")
            f.write(f"  Title: {title}\n")
            f.write(f"  Links: {len(links)}\n")
            f.write("-" * 50 + "\n")

        yield item

        # Follow links (basic depth limit via seen set in real version)
        for link in links[:10]:  # Limit to avoid explosion
            yield Request(link, callback=self.parse)

def run_crawler(urls):
    settings = get_project_settings()
    settings.update({
        'LOG_LEVEL': 'INFO',
    })
    process = CrawlerProcess(settings)
    process.crawl(SimpleSpider, start_urls=urls)
    process.start()

if __name__ == "__main__":
    cprint("kArmas_fram3work MINIMAL - Starting in Termux!", Colors.GREEN)
    start_urls = ['https://example.com']  # Change or input via args
    run_crawler(start_urls) 

# kArmas_fram3work.py
# HACKER MODE: ACTIVATED
# made in l0ve by kArmasec


import os
import re
import json
import hashlib
import logging
import asyncio
import threading
import subprocess
import signal
import sys
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
from enum import Enum
from contextlib import contextmanager
import io
import cProfile
import pstats
from datetime import datetime
from pathlib import Path
import jsonlines
import yaml
import click
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import websockets
import psutil
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import torch

# === HACKER COLORS ===
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    MATRIX = "\033[38;5;82m"
    PHOTON = "\033[38;5;159m"
    LOVE = "\033[38;5;198m"

def cprint(text: str, color: str = Colors.GREEN, end="\n"):
    print(f"{color}{text}{Colors.RESET}", end=end)

# === CORE LIBS ===
import scrapy
from scrapy import signals
from scrapy.http import HtmlResponse
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from bs4 import BeautifulSoup
from lxml import html as lxml_html, etree
from readability import Document
import chardet
from fake_useragent import UserAgent
from scrapy_playwright.page import PageMethod

# === PHOTON: Quantum AI Layer ===
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    PHOTON_AVAILABLE = True
    simulator = AerSimulator()
    cprint("PHOTON QUANTUM ENGINE: ONLINE", Colors.PHOTON)
except Exception as e:
    PHOTON_AVAILABLE = False
    cprint(f"PHOTON OFFLINE: {e}", Colors.RED)

# === AI & RAG ===
try:
    from transformers import pipeline
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=0 if torch.cuda.is_available() else -1)
    AI_AVAILABLE = True
    cprint("AI MODULES: LOADED", Colors.CYAN)
except Exception as e:
    AI_AVAILABLE = False
    cprint(f"AI ERROR: {e}", Colors.RED)

# === VECTOR DB (Chroma) ===
try:
    import chromadb
    from chromadb.utils import embedding_functions
    chroma_client = chromadb.PersistentClient(path="vector_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_or_create_collection(name="pages", embedding_function=embedding_fn)
    VECTOR_DB = True
    cprint("VECTOR DB: ACTIVE", Colors.MAGENTA)
except:
    VECTOR_DB = False
    cprint("VECTOR DB: OFFLINE", Colors.YELLOW)

# === STORAGE ===
from pymongo import MongoClient
import redis

# === PDF & EXPORT ===
from weasyprint import HTML as WeasyHTML

# === OUTPUT FILE ===
KARMAS_OUTPUT_FILE = Path("kArmasOutPut.txt")
KARMAS_OUTPUT_FILE.write_text(f"""{Colors.LOVE}
██╗  ██╗ █████╗ ██████╗ ███╗   ███╗ █████╗ ███████╗███████╗ ██████╗
██║ ██╔╝██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔════╝██╔════╝██╔════╝
█████╔╝ ███████║██████╔╝██╔████╔██║███████║███████╗█████╗  ██║     
██╔═██╗ ██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║╚════██║██╔══╝  ██║     
██║  ██╗██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║███████║███████╗╚██████╗
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝
{Colors.RESET}
made in l0ve by kArmasec
PHOTON QUANTUM AI | HACKER THEME | COLOR OUTPUT
{"="*70}

""")

# === CONFIG & PLUGINS ===
CONFIG_FILE = Path("karmas_config.yaml")
PLUGIN_DIR = Path("plugins")
PLUGIN_DIR.mkdir(exist_ok=True)

# === GLOBALS ===
ua = UserAgent()
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PDF_DIR = OUTPUT_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# === PROMETHEUS ===
pages_scraped = Counter('karmas_pages_scraped_total', 'Total pages scraped')
pages_failed = Counter('karmas_pages_failed_total', 'Failed pages')
latency_hist = Histogram('karmas_request_latency_seconds', 'Request latency')
cpu_gauge = Gauge('karmas_cpu_percent', 'CPU usage')
mem_gauge = Gauge('karmas_memory_percent', 'Memory usage')

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(level)s | %(message)s')
logger = logging.getLogger("kArmas_fram3work")

# =============================================================================
# PHOTON QUANTUM AI
# =============================================================================

def photon_quantum_enhance(text: str) -> Dict:
    if not PHOTON_AVAILABLE:
        return {"photon_score": 0.0, "quantum_label": "offline"}
    try:
        qc = QuantumCircuit(3)
        qc.h([0,1,2])
        qc.measure_all()
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=1).result()
        counts = result.get_counts()
        bitstring = list(counts.keys())[0]
        score = sum(int(b) for b in bitstring) / 3.0
        label = "POSITIVE" if score > 0.6 else "NEGATIVE" if score < 0.4 else "NEUTRAL"
        return {"photon_score": score, "quantum_label": label}
    except:
        return {"photon_score": 0.0, "quantum_label": "ERROR"}

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    "mongo_uri": "mongodb://localhost:27017",
    "redis_uri": "redis://localhost:6379/0",
    "parse_mode": "AGGRESSIVE",
    "use_playwright": True,
    "max_depth": 3,
    "max_pages": 1000,
    "delay": 1.0,
    "proxies": [],
    "plugins": ["ai", "pdf", "vector", "photon"],
    "api_port": 8000,
    "dashboard_port": 9000
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        return yaml.safe_load(CONFIG_FILE.read_text()) or DEFAULT_CONFIG
    CONFIG_FILE.write_text(yaml.dump(DEFAULT_CONFIG))
    return DEFAULT_CONFIG

config = load_config()

# =============================================================================
# CORE PARSER
# =============================================================================

class ParseMode(Enum):
    STRICT = "strict"
    LENIENT = "lenient"
    AGGRESSIVE = "aggressive"
    READABILITY = "readability"

class kArmasParser:
    def __init__(self, mode: ParseMode):
        self.mode = mode
        self.soup = None
        self.tree = None
        self.doc = None

    def parse(self, response: HtmlResponse) -> 'kArmasParser':
        text = response.text
        if isinstance(response.body, bytes):
            enc = chardet.detect(response.body)['encoding'] or 'utf-8'
            text = response.body.decode(enc, errors='replace')
        text = self._repair_html(text)
        self.soup = BeautifulSoup(text, 'html.parser')
        try:
            self.tree = lxml_html.fromstring(text)
        except:
            self.tree = lxml_html.fromstring(self.soup.prettify())
        if self.mode == ParseMode.READABILITY:
            self.doc = Document(text)
        return self

    def _repair_html(self, html: str) -> str:
        if self.mode == ParseMode.AGGRESSIVE:
            html = re.sub(r'<([^>]*)(?<!/)>', r'<\1/>', html)
            html = re.sub(r'&(?![a-zA-Z0-9#]+;)', '&amp;', html)
            stack = []
            for t in re.findall(r'<([^/\s>]+)', html):
                tag = t.lower().split()[0]
                if tag not in ['br','hr','img','input','meta','link']:
                    stack.append(tag)
            for t in reversed(re.findall(r'</([^>]+)>', html)):
                if stack and stack[-1] == t.lower():
                    stack.pop()
            for t in reversed(stack):
                html += f'</{t}>'
        return html.strip()

    def extract(self) -> dict:
        text = self.doc.summary() if self.mode == ParseMode.READABILITY and self.doc else self.soup.get_text(separator=' ', strip=True)
        title = self.doc.title() if self.doc else (self.soup.find('title').get_text(strip=True) if self.soup.find('title') else "")
        links = [{'href': urljoin(response.url, a['href']), 'text': a.get_text(strip=True)} for a in self.soup.find_all('a', href=True)]
        return {"title": title, "text": text, "links": links}

# =============================================================================
# PLUGIN SYSTEM
# =============================================================================

PLUGINS = {}

def plugin(name: str):
    def decorator(func):
        PLUGINS[name] = func
        return func
    return decorator

@plugin("ai")
def ai_plugin(data: dict):
    if not AI_AVAILABLE: return data
    data['classification'] = classifier(data['text'][:1000])[0] if data['text'] else {"label": "N/A", "score": 0.0}
    data['summary'] = summarizer(data['text'][:2000], max_length=150, min_length=50, do_sample=False)[0]['summary_text'] if len(data['text']) > 200 else data['text'][:500]
    return data

@plugin("pdf")
def pdf_plugin(data: dict):
    path = PDF_DIR / f"{uuid.uuid4().hex}.pdf"
    WeasyHTML(string=f"<h1>{data['title']}</h1><small>{data['url']}</small><hr>{data['sanitized']}").write_pdf(path)
    data['pdf_path'] = str(path)
    return data

@plugin("vector")
def vector_plugin(data: dict):
    if not VECTOR_DB: return data
    emb = embedder(data['text'][:1000])[0][0]
    collection.add(ids=[data['id']], documents=[data['text'][:2000]], metadatas=[{"url": data['url'], "title": data['title']}], embeddings=[emb])
    data['embedding'] = emb
    return data

@plugin("photon")
def photon_plugin(data: dict):
    result = photon_quantum_enhance(data['text'])
    data.update(result)
    return data

# =============================================================================
# SPIDER
# =============================================================================

class Fram3workSpider(scrapy.Spider):
    name = "fram3work"
    custom_settings = {
        'USER_AGENT': ua.random,
        'DOWNLOAD_DELAY': config['delay'],
        'CONCURRENT_REQUESTS': 16,
        'ROBOTSTXT_OBEY': True,
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 30000,
        'ITEM_PIPELINES': {'kArmas_fram3work.Fram3workPipeline': 300},
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.robotstxt.RobotsTxtMiddleware': 100,
            'kArmas_fram3work.RandomUAMiddleware': 400,
            'kArmas_fram3work.DedupeMiddleware': 420
        }
    }

    def __init__(self, start_urls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls or ['https://example.com']
        self.redis = redis.from_url(config['redis_uri'])
        self.mongo = MongoClient(config['mongo_uri'])
        self.col = self.mongo.karmas.pages
        for url in self.start_urls:
            self.redis.sadd("fram3work:queue", url)

    def start_requests(self):
        while self.redis.scard("fram3work:queue"):
            url = self.redis.spop("fram3work:queue")
            yield scrapy.Request(url, meta={'playwright': config['use_playwright']})

    async def parse(self, response: HtmlResponse):
        start = datetime.now()
        parser = kArmasParser(ParseMode[config['parse_mode']]).parse(response)
        data = parser.extract()
        item = {
            'id': str(uuid.uuid4()),
            'url': response.url,
            'title': data['title'],
            'text': data['text'],
            'links': data['links'][:50],
            'sanitized': parser.soup.prettify()[:50000],
            'crawled_at': datetime.utcnow().isoformat(),
            'domain': urlparse(response.url).netloc
        }

        # Apply plugins
        for plugin_name in config['plugins']:
            if plugin_name in PLUGINS:
                item = PLUGINS[plugin_name](item)

        # === COLORFUL OUTPUT TO CONSOLE ===
        cprint(f"\n[{Colors.BOLD}{Colors.GREEN}HACKED{Colors.RESET}] {Colors.CYAN}{item['url'][:60]}{Colors.RESET}", Colors.MATRIX)
        cprint(f"  Title: {Colors.YELLOW}{item['title'][:70]}{Colors.RESET}")
        if 'summary' in item:
            cprint(f"  Summary: {Colors.BLUE}{item['summary'][:100]}...{Colors.RESET}")
        if 'classification' in item:
            label = item['classification'].get('label', 'N/A')
            score = item['classification'].get('score', 0.0)
            color = Colors.GREEN if 'pos' in label.lower() else Colors.RED if 'neg' in label.lower() else Colors.YELLOW
            cprint(f"  AI: {color}{label.upper()} ({score:.2f}){Colors.RESET}")
        if 'quantum_label' in item:
            q_label = item['quantum_label']
            q_score = item['photon_score']
            q_color = Colors.PHOTON if q_label == "POSITIVE" else Colors.RED if q_label == "NEGATIVE" else Colors.YELLOW
            cprint(f"  PHOTON: {q_color}{q_label} ({q_score:.2f}){Colors.RESET}")
        cprint(f"  Links: {Colors.MAGENTA}{len(item['links'])}{Colors.RESET} | PDF: {Colors.GREEN}SAVED{Colors.RESET}")

        # === WRITE TO kArmasOutPut.txt ===
        with KARMAS_OUTPUT_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] HACKED: {item['url']}\n")
            f.write(f"  Title: {item['title']}\n")
            f.write(f"  AI: {item.get('classification', {}).get('label', 'N/A')} | PHOTON: {item.get('quantum_label', 'N/A')}\n")
            f.write(f"  Links: {len(item['links'])} | PDF: {item.get('pdf_path', 'N/A')}\n")
            f.write("-" * 70 + "\n")

        latency_hist.observe((datetime.now() - start).total_seconds())
        pages_scraped.inc()
        yield item

        for link in data['links']:
            if not any(x in link['href'] for x in ['mailto', 'javascript', '#']):
                self.redis.sadd("fram3work:queue", link['href'])

# =============================================================================
# PIPELINES & MIDDLEWARES
# =============================================================================

class Fram3workPipeline:
    def process_item(self, item, spider):
        return item

class RandomUAMiddleware:
    def process_request(self, request, spider):
        request.headers['User-Agent'] = ua.random

class DedupeMiddleware:
    def __init__(self):
        self.redis = redis.from_url(config['redis_uri'])
    def process_request(self, request, spider):
        if self.redis.sismember("fram3work:seen", request.url):
            return scrapy.http.Response(url=request.url, status=304)
        self.redis.sadd("fram3work:seen", request.url)

# =============================================================================
# FASTAPI + HACKER DASHBOARD
# =============================================================================

app = FastAPI(title="kArmas_fram3work — HACKER MODE")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return f"""
    <pre style="background:#000;color:#0f0;font-family:monospace;">
{Colors.MATRIX}
██╗  ██╗ █████╗ ██████╗ ███╗   ███╗ █████╗ ███████╗███████╗ ██████╗
██║ ██╔╝██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔════╝██╔════╝██╔════╝
█████╔╝ ███████║██████╔╝██╔████╔██║███████║███████╗█████╗  ██║     
██╔═██╗ ██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║╚════██║██╔══╝  ██║     
██║  ██╗██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║███████║███████╗╚██████╗
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝
{Colors.RESET}
<span style="color:#ff1493;">made in l0ve by kArmasec</span>
<span style="color:#00ffff;">PHOTON QUANTUM AI ACTIVE</span>

<span id="stats">Loading...</span>
<a href="/kArmasOutPut.txt" style="color:#0f0;">kArmasOutPut.txt</a>

<script>
setInterval(() => {{
  fetch('/api/stats').then(r => r.json()).then(d => {{
    document.getElementById('stats').innerHTML = `
<span style="color:#0f0;">HACKED: ${d.scraped}</span> | 
<span style="color:#f00;">FAILED: ${d.failed}</span> | 
<span style="color:#ff0;">CPU: ${d.cpu}%</span> | 
<span style="color:#0ff;">RAM: ${d.mem}%</span>
    `;
  }});
}}, 1000);
</script>
    </pre>
    """

@app.get("/kArmasOutPut.txt", response_class=PlainTextResponse)
async def get_output_file():
    return KARMAS_OUTPUT_FILE.read_text(encoding="utf-8")

@app.get("/api/stats")
async def api_stats():
    return {
        "scraped": int(pages_scraped._value()),
        "failed": int(pages_failed._value()),
        "cpu": psutil.cpu_percent(),
        "mem": psutil.virtual_memory().percent
    }

# =============================================================================
# CLI
# =============================================================================

@click.group()
def cli():
    pass

@cli.command()
@click.argument("urls", nargs=-1)
def hack(urls):
    """HACK THE PLANET"""
    cprint(f"\n{Colors.BOLD}{Colors.GREEN}INITIATING HACK SEQUENCE...{Colors.RESET}", Colors.MATRIX)
    run_framework(list(urls))

@cli.command()
def matrix():
    """ENTER THE MATRIX"""
    uvicorn.run(app, host="0.0.0.0", port=config['dashboard_port'])

# =============================================================================
# FRAMEWORK RUNNER
# =============================================================================

def run_framework(start_urls: List[str]):
    threading.Thread(target=start_http_server, args=(8000,), daemon=True).start()
    threading.Thread(target=uvicorn.run, kwargs={"app": app, "port": config['dashboard_port']}, daemon=True).start()

    settings = get_project_settings()
    settings.update({
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {'headless': True}
    })
    process = CrawlerProcess(settings)
    process.crawl(Fram3workSpider, start_urls=start_urls)
    process.start()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cprint(f"""
{Colors.MATRIX}
╔═══════════════════════════════════════════════════════════════╗
║               kArmas_fram3work — HACKER MODE                  ║
║                  made in l0ve by kArmasec                     ║
║                  PHOTON QUANTUM AI ACTIVE                     ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.RESET}
""", Colors.MATRIX)
    generate_requirements()
    generate_docker()
    cli()

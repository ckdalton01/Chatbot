# scraper.py
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re
import time
from typing import List, Dict, Set
from collections import defaultdict
from random import uniform
import logging

logging.basicConfig(
    filename="C:\Logs\mylog.log",
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class RobotsTxtParser:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.rules = defaultdict(list)
        self.sitemaps = []
        self.crawl_delay = 1.0  # Default delay
        self._parse_robots_txt()

    def _parse_robots_txt(self):
        robots_url = urljoin(self.base_url, '/robots.txt')
        try:
            response = requests.get(robots_url, timeout=5)
            if response.status_code == 200:
                logging.info(f"robots.txt found")
                self._process_robots_content(response.text)
        except requests.RequestException as e:
            logging.error(f"Warning: Could not fetch robots.txt: {e}")

    def _process_robots_content(self, content: str):
        current_user_agent = '*'
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower().startswith('user-agent:'):
                current_user_agent = line[11:].strip()
                continue
            if line.lower().startswith(('disallow:', 'allow:')):
                directive, path = line.split(':', 1)
                directive = directive.strip().lower()
                path = path.strip()
                pattern = self._path_to_regex(path)
                self.rules[current_user_agent].append((directive, pattern))
            elif line.lower().startswith('crawl-delay:'):
                try:
                    self.crawl_delay = float(line[12:].strip())
                except ValueError:
                    pass
            elif line.lower().startswith('sitemap:'):
                sitemap_url = line[8:].strip()
                self.sitemaps.append(sitemap_url)

    @staticmethod
    def _path_to_regex(path: str) -> str:
        if not path:
            return '^/$'
        path = re.escape(path)
        path = path.replace(r'\*', '.*').replace(r'\?', r'\?')
        if not path.startswith('^'):
            path = '^' + path
        if not path.endswith('$'):
            path = path + ('' if path.endswith('/') else '/?') + '$'
        return path

    def is_allowed(self, url: str, user_agent: str = '*') -> bool:
        parsed = urlparse(url)
        if parsed.netloc != urlparse(self.base_url).netloc:
            return False
        path = parsed.path or '/'
        for agent in [user_agent, '*']:
            if agent in self.rules:
                for directive, pattern in self.rules[agent]:
                    if re.match(pattern, path):
                        return directive == 'allow'
        return True

    def get_sitemap_urls(self) -> List[str]:
        urls = []
        for sitemap_url in self.sitemaps:
            try:
                response = requests.get(sitemap_url, timeout=5)
                if response.status_code == 200:
                    if 'sitemapindex' in response.text.lower():
                        urls.extend(self._parse_sitemap_index(response.text))
                    else:
                        urls.extend(self._parse_urlset_sitemap(response.text))
            except requests.RequestException as e:
                logging.critical(f"Error processing sitemap {sitemap_url}: {e}")
        return urls

    def _parse_sitemap_index(self, xml_content: str) -> List[str]:
        urls = []
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        for sitemap in soup.find_all('sitemap'):
            loc = sitemap.find('loc')
            if loc and loc.text:
                urls.extend(self._parse_urlset_sitemap(requests.get(loc.text).text))
        return urls

    def _parse_urlset_sitemap(self, xml_content: str) -> List[str]:
        urls = []
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        for url in soup.find_all('url'):
            loc = url.find('loc')
            if loc and loc.text and self.is_allowed(loc.text):
                urls.append(loc.text)
        return urls


class PoliteWebScraper:
    def __init__(self, base_url: str, user_agent: str = 'MyDocBot/1.0'):
        self.base_url = base_url
        self.user_agent = user_agent
        self.robots = RobotsTxtParser(base_url)
        self.visited = set()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

        self.skip_extensions = (
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.tiff',
            '.mp3', '.wav', '.ogg', '.mp4', '.mov', '.avi', '.mkv', '.webm',
            '.zip', '.rar', '.tar', '.gz', '.7z', '.exe', '.bin', '.dll', '.msi',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.js', '.css', '.woff', '.woff2', '.ttf', '.otf'
        )

        self.skip_patterns = [
            '/wp-content/uploads/',
            '/assets/',
            '/static/',
            '?attachment_id=',
            '~gitbook',
            '/pdf',
            '?only=yes',
            '&limit=',
            '&page=',
            '/tag/',
            '/catalog-release/'
            '/page/'
        ]

        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=self.robots.crawl_delay,
            status_forcelist=[408, 429, 500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _should_skip_url(self, url: str) -> bool:
        lower_url = url.lower()
        return lower_url.endswith(self.skip_extensions) or any(p in lower_url for p in self.skip_patterns)

    def scrape_site(self, max_pages: int = 100) -> Dict[str, str]:
        results = {}
        urls_to_visit = set(self.robots.get_sitemap_urls())

        if self.robots.is_allowed(self.base_url):
            urls_to_visit.add(self.base_url)

        while urls_to_visit and len(results) < max_pages:
            url = urls_to_visit.pop()

            if (
                url in self.visited or
                not self.robots.is_allowed(url) or
                self._should_skip_url(url)
            ):
                continue

            try:
                time.sleep(uniform(0.5, 1.5))
                logging.info(f"Fetching: {url}")
                response = self.session.get(url, timeout=10)
                time.sleep(self.robots.crawl_delay)

                if response.status_code == 200:
                    content = self._extract_content(response.text)
                    if not content:
                        continue
                    results[url] = content
                    self.visited.add(url)

                    new_links = self._extract_links(response.text, url)
                    for link in new_links:
                        if (
                            link not in self.visited and
                            link not in urls_to_visit and
                            self.robots.is_allowed(link) and
                            not self._should_skip_url(link)
                        ):
                            urls_to_visit.add(link)

            except requests.RequestException as e:
                logging.error(f"Error fetching {url}: {e}")

        return results

    def _extract_content(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()

        text_elements = []
        for el in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            if el.get_text(strip=True):
                # Skip if element has mostly links (e.g., <p><a>...</a><a>...</a></p>)
                links = el.find_all('a')
                total_children = len(list(el.children))
                if links and len(links) >= total_children - 1:
                    continue
                text_elements.append(el.get_text(strip=True))
        text = ' '.join(text_elements)

        num_links = len(soup.find_all('a'))
        if len(text) < 50 and num_links > 10:
            return ''

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            full_url = full_url.split('#')[0]
            if urlparse(full_url).netloc == urlparse(self.base_url).netloc:
                if not self._should_skip_url(full_url):
                    links.add(full_url)
        return links

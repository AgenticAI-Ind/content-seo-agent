"""
Web Scraping Utilities
Tools for competitor analysis and content research.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
from loguru import logger
import time


class WebScraper:
    """
    Web scraping utility for content research.

    Features:
    - HTML content extraction
    - Metadata extraction
    - Competitor analysis
    - SEO data gathering
    """

    def __init__(self, user_agent: str = None):
        """
        Initialize web scraper.

        Args:
            user_agent: Custom user agent string
        """
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }

        logger.info("WebScraper initialized")

    def scrape_url(
        self,
        url: str,
        timeout: int = 10
    ) -> Dict[str, Any]:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape
            timeout: Request timeout in seconds

        Returns:
            Dictionary with scraped content and metadata
        """
        logger.info(f"Scraping URL: {url}")

        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            return {
                "url": url,
                "success": True,
                "title": self._extract_title(soup),
                "meta_description": self._extract_meta_description(soup),
                "headings": self._extract_headings(soup),
                "content": self._extract_main_content(soup),
                "links": self._extract_links(soup, url),
                "images": self._extract_images(soup, url),
                "word_count": len(self._extract_text(soup).split()),
                "keywords": self._extract_meta_keywords(soup)
            }

        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.text.strip() if title_tag else None

    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content', '').strip() if meta_desc else None

    def _extract_meta_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract meta keywords"""
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})

        if meta_keywords:
            keywords = meta_keywords.get('content', '')
            return [k.strip() for k in keywords.split(',') if k.strip()]

        return []

    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all headings (H1-H6)"""
        headings = {}

        for level in range(1, 7):
            tag = f'h{level}'
            headings[tag] = [h.text.strip() for h in soup.find_all(tag)]

        return headings

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content"""
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Try common content containers
        content_tags = ['article', 'main', 'div.content', 'div.post']

        for tag in content_tags:
            content = soup.find(tag.split('.')[0], class_=tag.split('.')[1] if '.' in tag else None)
            if content:
                return content.get_text(separator='\n', strip=True)

        # Fallback to body
        body = soup.find('body')
        return body.get_text(separator='\n', strip=True) if body else ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract all text"""
        return soup.get_text(separator=' ', strip=True)

    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str
    ) -> List[Dict[str, str]]:
        """Extract all links"""
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(base_url, href)

            links.append({
                "url": absolute_url,
                "text": a_tag.text.strip(),
                "is_external": urlparse(absolute_url).netloc != urlparse(base_url).netloc
            })

        return links

    def _extract_images(
        self,
        soup: BeautifulSoup,
        base_url: str
    ) -> List[Dict[str, str]]:
        """Extract all images"""
        images = []

        for img_tag in soup.find_all('img'):
            src = img_tag.get('src', '')
            if src:
                absolute_url = urljoin(base_url, src)

                images.append({
                    "url": absolute_url,
                    "alt": img_tag.get('alt', ''),
                    "title": img_tag.get('title', '')
                })

        return images

    def analyze_competitor(self, url: str) -> Dict[str, Any]:
        """
        Perform competitor content analysis.

        Args:
            url: Competitor URL

        Returns:
            Analysis results
        """
        logger.info(f"Analyzing competitor: {url}")

        data = self.scrape_url(url)

        if not data.get('success'):
            return data

        # Additional analysis
        analysis = {
            "url": url,
            "title": data['title'],
            "meta_description": data['meta_description'],
            "word_count": data['word_count'],
            "heading_count": {
                h: len(data['headings'].get(h, []))
                for h in ['h1', 'h2', 'h3']
            },
            "internal_links": sum(1 for l in data['links'] if not l['is_external']),
            "external_links": sum(1 for l in data['links'] if l['is_external']),
            "image_count": len(data['images']),
            "images_with_alt": sum(1 for img in data['images'] if img['alt']),
            "keywords": data['keywords']
        }

        return analysis

    def batch_scrape(
        self,
        urls: List[str],
        delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs with rate limiting.

        Args:
            urls: List of URLs to scrape
            delay: Delay between requests (seconds)

        Returns:
            List of scraping results
        """
        logger.info(f"Batch scraping {len(urls)} URLs")

        results = []

        for i, url in enumerate(urls):
            logger.debug(f"Scraping {i+1}/{len(urls)}: {url}")

            result = self.scrape_url(url)
            results.append(result)

            # Rate limiting
            if i < len(urls) - 1:
                time.sleep(delay)

        return results


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()

    # Scrape a URL
    url = "https://example.com"
    result = scraper.scrape_url(url)

    if result['success']:
        print(f"Title: {result['title']}")
        print(f"Word Count: {result['word_count']}")
        print(f"Headings: {result['headings']}")
    else:
        print(f"Error: {result['error']}")

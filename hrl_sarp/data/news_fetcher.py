"""
File: news_fetcher.py
Module: data
Description: RSS/web scraper for Indian financial news from Economic Times, Moneycontrol,
             and BSE announcements. Provides raw news text for FinBERT sentiment scoring.
Design Decisions:
    - RSS feeds for structured extraction; web scraping as fallback for full articles.
    - feedparser for RSS; BeautifulSoup for HTML article content.
    - Rate limiting to respect server policies and avoid IP bans.
References:
    - feedparser: https://feedparser.readthedocs.io/
    - Economic Times RSS, Moneycontrol RSS, BSE India API
Author: HRL-SARP Framework
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class NewsArticle:
    """Structured representation of a financial news article."""

    def __init__(
        self, title: str, content: str, source: str, category: str,
        published_at: datetime, url: str, symbols: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
    ) -> None:
        self.title = title
        self.content = content
        self.source = source
        self.category = category
        self.published_at = published_at
        self.url = url
        self.symbols = symbols or []
        self.sectors = sectors or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title, "content": self.content, "source": self.source,
            "category": self.category, "published_at": self.published_at.isoformat(),
            "url": self.url, "symbols": self.symbols, "sectors": self.sectors,
        }


class NewsFetcher:
    """
    Fetches financial news from Indian sources via RSS and web scraping.

    Sources: Economic Times Markets, Moneycontrol, BSE announcements, LiveMint.
    Returns structured NewsArticle objects with title, full content, source metadata,
    and extracted entity/sector tags for downstream sentiment analysis by FinBERT-India.
    """

    # ── NSE sector keywords for tagging ──────────────────────────────
    SECTOR_KEYWORDS: Dict[str, List[str]] = {
        "IT": ["software", "infosys", "tcs", "wipro", "hcl", "tech mahindra", "it sector", "technology"],
        "Financials": ["bank", "nbfc", "insurance", "nifty bank", "hdfc", "icici", "kotak", "bajaj finance"],
        "Auto": ["automobile", "maruti", "tata motors", "mahindra", "bajaj auto", "ev", "electric vehicle"],
        "Pharma": ["pharma", "drug", "fda", "sun pharma", "dr reddy", "cipla", "healthcare", "hospital"],
        "FMCG": ["fmcg", "consumer goods", "hindustan unilever", "itc", "nestle", "britannia", "dabur"],
        "Energy": ["oil", "gas", "reliance", "ongc", "bpcl", "ioc", "power", "ntpc", "energy"],
        "Metals": ["steel", "metal", "tata steel", "hindalco", "jsw", "coal", "mining", "vedanta"],
        "Realty": ["real estate", "realty", "dlf", "godrej properties", "housing", "property"],
        "Media": ["media", "entertainment", "zee", "sun tv", "pvr", "streaming"],
        "Telecom": ["telecom", "bharti airtel", "jio", "vodafone", "5g", "spectrum"],
        "Infra": ["infrastructure", "larsen", "l&t", "cement", "construction", "highway", "road"],
    }

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        self.news_config = self.config["news"]
        self.scraping_config = self.news_config["scraping"]
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.scraping_config.get("user_agent", "HRL-SARP-Research/1.0"),
        })
        logger.info("NewsFetcher initialised | sources=%d", len(self.news_config["sources"]))

    # ══════════════════════════════════════════════════════════════════
    # MAIN FETCH METHOD
    # ══════════════════════════════════════════════════════════════════
    def fetch_all_news(self, lookback_hours: Optional[int] = None) -> List[NewsArticle]:
        """
        Fetch news from all configured sources.
        Args:
            lookback_hours: Override config lookback window (in hours).
        Returns:
            List of NewsArticle objects sorted by publication date (newest first).
        """
        lookback = lookback_hours or self.scraping_config.get("lookback_hours", 168)
        cutoff = datetime.now() - timedelta(hours=lookback)
        all_articles: List[NewsArticle] = []

        for source_cfg in self.news_config["sources"]:
            try:
                if source_cfg["type"] == "rss":
                    articles = self._fetch_rss(source_cfg, cutoff)
                elif source_cfg["type"] == "api":
                    articles = self._fetch_bse_announcements(source_cfg, cutoff)
                else:
                    continue
                all_articles.extend(articles)
                logger.info("Fetched %d articles from %s", len(articles), source_cfg["name"])
                time.sleep(self.scraping_config.get("rate_limit_seconds", 3))
            except Exception as e:
                logger.warning("Failed to fetch from %s: %s", source_cfg["name"], e)

        # Sort by date, newest first
        all_articles.sort(key=lambda a: a.published_at, reverse=True)
        logger.info("Total articles fetched: %d", len(all_articles))
        return all_articles

    # ══════════════════════════════════════════════════════════════════
    # RSS FEED FETCHING
    # ══════════════════════════════════════════════════════════════════
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_rss(self, source_cfg: Dict[str, str], cutoff: datetime) -> List[NewsArticle]:
        """Parse RSS feed and extract articles published after cutoff."""
        feed = feedparser.parse(source_cfg["url"])
        articles: List[NewsArticle] = []
        max_articles = self.scraping_config.get("max_articles_per_source", 100)

        for entry in feed.entries[:max_articles]:
            try:
                # Parse publication date
                pub_date = self._parse_rss_date(entry)
                if pub_date and pub_date < cutoff:
                    continue

                # Extract content
                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                content = BeautifulSoup(summary, "html.parser").get_text(strip=True)
                url = entry.get("link", "")

                # Tag sectors and symbols
                full_text = f"{title} {content}".lower()
                sectors = self._tag_sectors(full_text)
                symbols = self._extract_symbols(full_text)

                article = NewsArticle(
                    title=title, content=content, source=source_cfg["name"],
                    category=source_cfg.get("category", "markets"),
                    published_at=pub_date or datetime.now(), url=url,
                    symbols=symbols, sectors=sectors,
                )
                articles.append(article)
            except Exception as e:
                logger.debug("Skipping RSS entry: %s", e)
                continue
        return articles

    # ══════════════════════════════════════════════════════════════════
    # BSE ANNOUNCEMENTS
    # ══════════════════════════════════════════════════════════════════
    def _fetch_bse_announcements(self, source_cfg: Dict[str, str], cutoff: datetime) -> List[NewsArticle]:
        """Fetch corporate announcements from BSE India API."""
        try:
            headers = {"Accept": "application/json", "Referer": "https://www.bseindia.com"}
            params = {
                "strCat": "-1", "strPrevDate": cutoff.strftime("%Y%m%d"),
                "strScrip": "", "strSearch": "P", "strToDate": datetime.now().strftime("%Y%m%d"),
                "strType": "C",
            }
            response = self.session.get(source_cfg["url"], params=params, headers=headers, timeout=15)
            if response.status_code != 200:
                return []

            data = response.json()
            if not isinstance(data, dict) or "Table" not in data:
                return []

            articles: List[NewsArticle] = []
            for item in data["Table"][:self.scraping_config.get("max_articles_per_source", 100)]:
                try:
                    pub_str = item.get("NEWS_DT", "")
                    pub_date = datetime.strptime(pub_str, "%d/%m/%Y %H:%M:%S") if pub_str else datetime.now()
                    title = item.get("NEWSSUB", "")
                    content = item.get("HEADLINE", title)
                    symbol = item.get("SCRIP_CD", "")
                    company = item.get("SLONGNAME", "")

                    article = NewsArticle(
                        title=title, content=f"{company}: {content}",
                        source="BSE Announcements", category="corporate",
                        published_at=pub_date, url=item.get("NSURL", ""),
                        symbols=[symbol] if symbol else [],
                        sectors=self._tag_sectors(f"{title} {content}".lower()),
                    )
                    articles.append(article)
                except Exception:
                    continue
            return articles
        except Exception as e:
            logger.warning("BSE announcements fetch failed: %s", e)
            return []

    # ══════════════════════════════════════════════════════════════════
    # FULL ARTICLE SCRAPING
    # ══════════════════════════════════════════════════════════════════
    def scrape_article_content(self, url: str) -> str:
        """
        Scrape full article text from a URL. Used when RSS provides only summary.
        Returns cleaned article text suitable for FinBERT sentiment scoring.
        """
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            # Remove scripts, styles, nav, ads
            for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
                tag.decompose()

            # Try common article body selectors
            article_body = (
                soup.find("div", class_="artText") or  # Economic Times
                soup.find("div", class_="content_wrapper") or  # Moneycontrol
                soup.find("article") or
                soup.find("div", class_="story-element")  # LiveMint
            )

            if article_body:
                paragraphs = article_body.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
            else:
                text = soup.get_text(separator=" ", strip=True)

            # Clean text
            text = re.sub(r"\s+", " ", text).strip()
            return text[:5000]  # Cap at 5000 chars for NLP processing

        except Exception as e:
            logger.warning("Article scrape failed for %s: %s", url, e)
            return ""

    # ══════════════════════════════════════════════════════════════════
    # SECTOR AND SYMBOL TAGGING
    # ══════════════════════════════════════════════════════════════════
    def _tag_sectors(self, text: str) -> List[str]:
        """Tag article with relevant sectors based on keyword matching."""
        tagged: List[str] = []
        for sector, keywords in self.SECTOR_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                tagged.append(sector)
        return tagged

    @staticmethod
    def _extract_symbols(text: str) -> List[str]:
        """Extract potential NSE stock symbols mentioned in text."""
        # Common large-cap symbols to look for (simplified pattern matching)
        known_symbols = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
            "TATAMOTORS", "MARUTI", "SUNPHARMA", "ITC", "HINDUNILVR",
            "BAJFINANCE", "KOTAKBANK", "AXISBANK", "WIPRO", "HCLTECH",
            "TATASTEEL", "ONGC", "NTPC", "BHARTIARTL", "DLF",
        ]
        found = [s for s in known_symbols if s.lower() in text]
        return found

    # ══════════════════════════════════════════════════════════════════
    # CONVERT TO DATAFRAME
    # ══════════════════════════════════════════════════════════════════
    def articles_to_dataframe(self, articles: List[NewsArticle]) -> pd.DataFrame:
        """Convert list of NewsArticle to DataFrame for feature store."""
        if not articles:
            return pd.DataFrame()
        data = [a.to_dict() for a in articles]
        df = pd.DataFrame(data)
        df["published_at"] = pd.to_datetime(df["published_at"])
        df = df.set_index("published_at").sort_index(ascending=False)
        return df

    # ══════════════════════════════════════════════════════════════════
    # DATE PARSING UTILITY
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _parse_rss_date(entry: Any) -> Optional[datetime]:
        """Parse date from RSS entry, handling multiple formats."""
        date_str = entry.get("published", entry.get("updated", ""))
        if not date_str:
            return None
        date_formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",
            "%d %b %Y %H:%M:%S",
        ]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            except ValueError:
                continue
        return None

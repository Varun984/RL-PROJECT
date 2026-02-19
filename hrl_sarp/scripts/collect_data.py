"""
Data Collection Script for HRL-SARP
Collects historical market data, macro indicators, fundamentals, and news
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.market_data_fetcher import MarketDataFetcher
from data.macro_fetcher import MacroFetcher
from data.fundamental_fetcher import FundamentalFetcher
from data.news_fetcher import NewsFetcher
from utils.common import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect historical data for HRL-SARP")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--skip-market", action="store_true", help="Skip market data collection")
    parser.add_argument("--skip-macro", action="store_true", help="Skip macro data collection")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Skip fundamentals collection")
    parser.add_argument("--skip-news", action="store_true", help="Skip news collection")
    parser.add_argument("--sectors", type=str, nargs="+", help="Specific sectors to collect (default: all)")
    return parser.parse_args()


def collect_market_data(fetcher: MarketDataFetcher, start_date: str, end_date: str, output_dir: Path):
    """Collect OHLCV data for all stocks in the universe."""
    logger.info("=" * 70)
    logger.info("COLLECTING MARKET DATA")
    logger.info("=" * 70)
    
    # Get sector universe
    sector_universe = fetcher.get_sector_universe()
    all_symbols = fetcher.get_all_symbols()
    
    logger.info(f"Total symbols to fetch: {len(all_symbols)}")
    logger.info(f"Sectors: {list(sector_universe.keys())}")
    
    # Create output directory
    market_dir = output_dir / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data for each sector
    for sector, symbols in sector_universe.items():
        logger.info(f"\n📊 Fetching {sector} sector ({len(symbols)} stocks)...")
        
        try:
            # Batch fetch for efficiency
            sector_data_dict = fetcher.fetch_ohlcv_batch(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            # Convert dict to DataFrame
            all_data = []
            for symbol, df in sector_data_dict.items():
                if df is not None and not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
            
            if all_data:
                sector_data = pd.concat(all_data, ignore_index=True)
                
                # Save sector data
                sector_file = market_dir / f"{sector.lower()}_ohlcv.parquet"
                sector_data.to_parquet(sector_file)
                logger.info(f"✅ Saved {sector} data: {len(sector_data)} rows → {sector_file}")
            else:
                logger.warning(f"⚠️  No data collected for {sector}")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch {sector} data: {e}")
    
    # Fetch index data
    logger.info("\n📈 Fetching index data...")
    indices = {
        "NSEI": "^NSEI",      # Nifty 50
        "NSEBANK": "^NSEBANK", # Bank Nifty
        "CNXIT": "^CNXIT",     # IT Index
        "CNXAUTO": "^CNXAUTO"  # Auto Index
    }
    
    for name, symbol in indices.items():
        try:
            index_data = fetcher.fetch_ohlcv(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            if index_data is not None and not index_data.empty:
                index_file = market_dir / f"{name}_index.parquet"
                index_data.to_parquet(index_file)
                logger.info(f"✅ Saved {name} data: {len(index_data)} rows")
        except Exception as e:
            logger.error(f"❌ Failed to fetch {name}: {e}")
    
    logger.info(f"\n✅ Market data collection complete! Saved to {market_dir}")


def collect_macro_data(fetcher: MacroFetcher, start_date: str, end_date: str, output_dir: Path):
    """Collect macroeconomic indicators."""
    logger.info("=" * 70)
    logger.info("COLLECTING MACRO DATA")
    logger.info("=" * 70)
    
    macro_dir = output_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    
    # List of macro indicators to fetch
    indicators = {
        "gdp": "GDP Growth Rate",
        "inflation": "CPI Inflation",
        "interest_rate": "Repo Rate",
        "iip": "Industrial Production Index",
        "pmi_manufacturing": "Manufacturing PMI",
        "pmi_services": "Services PMI",
        "fii_flows": "FII Net Flows",
        "dii_flows": "DII Net Flows",
        "crude_oil": "Crude Oil Price",
        "usd_inr": "USD/INR Exchange Rate",
        "gold": "Gold Price",
        "vix": "India VIX",
    }
    
    all_macro_data = {}
    
    for indicator_key, indicator_name in indicators.items():
        logger.info(f"\n📊 Fetching {indicator_name}...")
        try:
            data = fetcher.fetch_indicator(indicator_key, start_date, end_date)
            if data is not None and not data.empty:
                all_macro_data[indicator_key] = data
                logger.info(f"✅ Fetched {indicator_name}: {len(data)} data points")
            else:
                logger.warning(f"⚠️  No data for {indicator_name}")
        except Exception as e:
            logger.error(f"❌ Failed to fetch {indicator_name}: {e}")
    
    # Combine all macro data
    if all_macro_data:
        combined_macro = pd.DataFrame(all_macro_data)
        macro_file = macro_dir / "macro_indicators.parquet"
        combined_macro.to_parquet(macro_file)
        logger.info(f"\n✅ Macro data saved: {macro_file}")
    else:
        logger.warning("⚠️  No macro data collected")


def collect_fundamental_data(fetcher: FundamentalFetcher, symbols: list, output_dir: Path):
    """Collect fundamental data for stocks."""
    logger.info("=" * 70)
    logger.info("COLLECTING FUNDAMENTAL DATA")
    logger.info("=" * 70)
    
    fundamentals_dir = output_dir / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Fetching fundamentals for {len(symbols)} stocks...")
    
    all_fundamentals = []
    
    for i, symbol in enumerate(symbols, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(symbols)} stocks...")
        
        try:
            fund_data = fetcher.fetch_fundamentals(symbol)
            if fund_data:
                fund_data["symbol"] = symbol
                fund_data["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
                all_fundamentals.append(fund_data)
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
    
    if all_fundamentals:
        fundamentals_df = pd.DataFrame(all_fundamentals)
        fund_file = fundamentals_dir / "fundamentals.parquet"
        fundamentals_df.to_parquet(fund_file)
        logger.info(f"\n✅ Fundamentals saved: {len(fundamentals_df)} stocks → {fund_file}")
    else:
        logger.warning("⚠️  No fundamental data collected")


def collect_news_data(fetcher: NewsFetcher, start_date: str, end_date: str, output_dir: Path):
    """Collect news articles."""
    logger.info("=" * 70)
    logger.info("COLLECTING NEWS DATA")
    logger.info("=" * 70)
    
    news_dir = output_dir / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    
    # News sources and keywords
    keywords = [
        "nifty", "sensex", "stock market", "india economy",
        "rbi", "inflation", "gdp", "fii", "dii"
    ]
    
    all_news = []
    
    for keyword in keywords:
        logger.info(f"\n📰 Fetching news for: {keyword}")
        try:
            news_articles = fetcher.fetch_news(
                query=keyword,
                start_date=start_date,
                end_date=end_date,
                max_articles=100
            )
            if news_articles:
                all_news.extend(news_articles)
                logger.info(f"✅ Fetched {len(news_articles)} articles")
        except Exception as e:
            logger.error(f"❌ Failed to fetch news for {keyword}: {e}")
    
    if all_news:
        news_df = pd.DataFrame(all_news)
        # Remove duplicates
        news_df = news_df.drop_duplicates(subset=["title", "published_date"])
        news_file = news_dir / "news_articles.parquet"
        news_df.to_parquet(news_file)
        logger.info(f"\n✅ News data saved: {len(news_df)} articles → {news_file}")
    else:
        logger.warning("⚠️  No news data collected")


def main():
    args = parse_args()
    
    # Setup logging
    log_dir = PROJECT_ROOT / "logs" / "data_collection"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.info("=" * 70)
    logger.info("HRL-SARP DATA COLLECTION")
    logger.info("=" * 70)
    logger.info(f"Start Date: {args.start}")
    logger.info(f"End Date: {args.end}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize fetchers
    try:
        market_fetcher = MarketDataFetcher() if not args.skip_market else None
        macro_fetcher = MacroFetcher() if not args.skip_macro else None
        fundamental_fetcher = FundamentalFetcher() if not args.skip_fundamentals else None
        news_fetcher = NewsFetcher() if not args.skip_news else None
    except Exception as e:
        logger.error(f"Failed to initialize fetchers: {e}")
        return
    
    # Collect market data
    if not args.skip_market and market_fetcher:
        try:
            collect_market_data(market_fetcher, args.start, args.end, output_dir)
        except Exception as e:
            logger.error(f"Market data collection failed: {e}", exc_info=True)
    
    # Collect macro data
    if not args.skip_macro and macro_fetcher:
        try:
            collect_macro_data(macro_fetcher, args.start, args.end, output_dir)
        except Exception as e:
            logger.error(f"Macro data collection failed: {e}", exc_info=True)
    
    # Collect fundamental data
    if not args.skip_fundamentals and fundamental_fetcher and market_fetcher:
        try:
            all_symbols = market_fetcher.get_all_symbols()
            collect_fundamental_data(fundamental_fetcher, all_symbols, output_dir)
        except Exception as e:
            logger.error(f"Fundamental data collection failed: {e}", exc_info=True)
    
    # Collect news data
    if not args.skip_news and news_fetcher:
        try:
            collect_news_data(news_fetcher, args.start, args.end, output_dir)
        except Exception as e:
            logger.error(f"News data collection failed: {e}", exc_info=True)
    
    logger.info("=" * 70)
    logger.info("DATA COLLECTION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Data saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Run feature engineering: python scripts/engineer_features.py")
    logger.info("2. Start training: python main.py train")


if __name__ == "__main__":
    main()

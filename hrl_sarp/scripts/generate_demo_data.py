"""
Generate Demo Data for HRL-SARP
Creates synthetic but realistic market data for testing
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import setup_logging

logger = logging.getLogger(__name__)

# Indian stock universe
SECTOR_UNIVERSE = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Financials": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS", "DIVISLAB.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS"],
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "NMDC.NS"],
    "Realty": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS"],
    "Media": ["ZEEL.NS", "SUNTV.NS", "PVR.NS", "DISHTV.NS"],
    "Infra": ["LT.NS", "ADANIPORTS.NS", "IRBINFRA.NS", "GMRINFRA.NS", "NBCC.NS"],
}


def generate_market_data(start_date: str, end_date: str, output_dir: Path):
    """Generate synthetic OHLCV data."""
    logger.info("Generating market data...")
    
    market_dir = output_dir / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    
    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    n_days = len(dates)
    
    # Generate data for each sector
    for sector, symbols in SECTOR_UNIVERSE.items():
        logger.info(f"  Generating {sector} sector...")
        
        sector_data = []
        
        for symbol in symbols:
            # Random walk with drift for price
            base_price = np.random.uniform(100, 2000)
            drift = np.random.uniform(-0.0002, 0.0008)  # Daily drift
            volatility = np.random.uniform(0.015, 0.035)  # Daily volatility
            
            returns = np.random.normal(drift, volatility, n_days)
            prices = base_price * np.cumprod(1 + returns)
            
            # Generate OHLCV
            for i, date in enumerate(dates):
                price = prices[i]
                daily_range = price * np.random.uniform(0.01, 0.03)
                
                open_price = price + np.random.uniform(-daily_range/2, daily_range/2)
                high_price = max(open_price, price) + np.random.uniform(0, daily_range/2)
                low_price = min(open_price, price) - np.random.uniform(0, daily_range/2)
                close_price = price
                volume = int(np.random.lognormal(15, 1.5))  # Log-normal volume
                
                sector_data.append({
                    "date": date,
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                })
        
        # Save sector data
        sector_df = pd.DataFrame(sector_data)
        sector_file = market_dir / f"{sector.lower()}_ohlcv.parquet"
        sector_df.to_parquet(sector_file, index=False)
        logger.info(f"    ✅ Saved {len(sector_df)} rows to {sector_file.name}")
    
    # Generate index data
    logger.info("  Generating index data...")
    index_data = []
    base_index = 18000
    
    for date in dates:
        daily_return = np.random.normal(0.0005, 0.012)
        base_index *= (1 + daily_return)
        daily_range = base_index * 0.015
        
        index_data.append({
            "date": date,
            "open": round(base_index + np.random.uniform(-daily_range, daily_range), 2),
            "high": round(base_index + np.random.uniform(0, daily_range), 2),
            "low": round(base_index - np.random.uniform(0, daily_range), 2),
            "close": round(base_index, 2),
            "volume": int(np.random.lognormal(20, 1)),
        })
    
    index_df = pd.DataFrame(index_data)
    index_file = market_dir / "NSEI_index.parquet"
    index_df.to_parquet(index_file, index=False)
    logger.info(f"    ✅ Saved {len(index_df)} rows to {index_file.name}")


def generate_macro_data(start_date: str, end_date: str, output_dir: Path):
    """Generate synthetic macro indicators."""
    logger.info("Generating macro data...")
    
    macro_dir = output_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    
    # Monthly data for macro indicators
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")  # Month start
    n_months = len(dates)
    
    macro_data = {
        "date": dates,
        "gdp_growth": np.random.uniform(5.5, 8.5, n_months),  # %
        "inflation": np.random.uniform(4.0, 7.0, n_months),  # %
        "interest_rate": np.random.uniform(5.5, 7.5, n_months),  # %
        "iip": np.random.uniform(-2, 10, n_months),  # %
        "pmi_manufacturing": np.random.uniform(48, 58, n_months),
        "pmi_services": np.random.uniform(50, 62, n_months),
        "fii_flows": np.random.normal(0, 5000, n_months),  # Crores
        "dii_flows": np.random.normal(0, 3000, n_months),  # Crores
        "crude_oil": np.random.uniform(60, 90, n_months),  # USD/barrel
        "usd_inr": np.random.uniform(74, 84, n_months),
        "gold": np.random.uniform(45000, 55000, n_months),  # INR/10g
        "vix": np.random.uniform(12, 25, n_months),
    }
    
    macro_df = pd.DataFrame(macro_data)
    macro_file = macro_dir / "macro_indicators.parquet"
    macro_df.to_parquet(macro_file, index=False)
    logger.info(f"  ✅ Saved {len(macro_df)} rows to {macro_file.name}")


def generate_fundamental_data(output_dir: Path):
    """Generate synthetic fundamental data."""
    logger.info("Generating fundamental data...")
    
    fundamentals_dir = output_dir / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    
    all_symbols = [symbol for symbols in SECTOR_UNIVERSE.values() for symbol in symbols]
    
    fundamental_data = []
    for symbol in all_symbols:
        fundamental_data.append({
            "symbol": symbol,
            "pe_ratio": np.random.uniform(10, 50),
            "pb_ratio": np.random.uniform(1, 10),
            "roe": np.random.uniform(5, 25),
            "debt_to_equity": np.random.uniform(0, 2),
            "current_ratio": np.random.uniform(0.8, 3),
            "market_cap": np.random.uniform(10000, 500000),  # Crores
            "dividend_yield": np.random.uniform(0, 4),  # %
            "eps": np.random.uniform(5, 100),
            "revenue_growth": np.random.uniform(-10, 30),  # %
            "profit_margin": np.random.uniform(5, 25),  # %
            "fetch_date": datetime.now().strftime("%Y-%m-%d"),
        })
    
    fundamentals_df = pd.DataFrame(fundamental_data)
    fund_file = fundamentals_dir / "fundamentals.parquet"
    fundamentals_df.to_parquet(fund_file, index=False)
    logger.info(f"  ✅ Saved {len(fundamentals_df)} rows to {fund_file.name}")


def generate_news_data(start_date: str, end_date: str, output_dir: Path):
    """Generate synthetic news data."""
    logger.info("Generating news data...")
    
    news_dir = output_dir / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    headlines = [
        "Markets rally on strong earnings",
        "FII inflows boost sentiment",
        "RBI maintains status quo on rates",
        "IT sector outperforms on dollar strength",
        "Banking stocks under pressure",
        "Pharma sector sees buying interest",
        "Auto sales disappoint in Q2",
        "Infrastructure spending to boost growth",
        "Inflation concerns weigh on markets",
        "Global cues drive market direction",
    ]
    
    news_data = []
    for date in dates:
        # 2-5 news articles per day
        n_articles = np.random.randint(2, 6)
        for _ in range(n_articles):
            news_data.append({
                "date": date,
                "title": np.random.choice(headlines),
                "sentiment": np.random.uniform(-1, 1),  # -1 to 1
                "source": np.random.choice(["Economic Times", "Mint", "Business Standard"]),
            })
    
    news_df = pd.DataFrame(news_data)
    news_file = news_dir / "news_articles.parquet"
    news_df.to_parquet(news_file, index=False)
    logger.info(f"  ✅ Saved {len(news_df)} rows to {news_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate demo data for HRL-SARP")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    # Setup logging
    log_dir = PROJECT_ROOT / "logs" / "data_generation"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / f"generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.info("=" * 70)
    logger.info("GENERATING DEMO DATA FOR HRL-SARP")
    logger.info("=" * 70)
    logger.info(f"Start Date: {args.start}")
    logger.info(f"End Date: {args.end}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all data types
    generate_market_data(args.start, args.end, output_dir)
    generate_macro_data(args.start, args.end, output_dir)
    generate_fundamental_data(output_dir)
    generate_news_data(args.start, args.end, output_dir)
    
    logger.info("=" * 70)
    logger.info("DEMO DATA GENERATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Data saved to: {output_dir}")
    logger.info("\nData summary:")
    logger.info(f"  - Market data: {len(SECTOR_UNIVERSE)} sectors, {sum(len(s) for s in SECTOR_UNIVERSE.values())} stocks")
    logger.info(f"  - Macro data: 12 indicators")
    logger.info(f"  - Fundamental data: {sum(len(s) for s in SECTOR_UNIVERSE.values())} stocks")
    logger.info(f"  - News data: Multiple articles per day")
    logger.info("\nNext steps:")
    logger.info("1. Run feature engineering: python scripts/engineer_features.py")
    logger.info("2. Start training: python main.py train")


if __name__ == "__main__":
    main()

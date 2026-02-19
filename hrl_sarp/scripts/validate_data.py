"""
Data Validation Script
Checks if collected data is correct and ready for training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def validate_market_data():
    """Validate market OHLCV data."""
    print("=" * 70)
    print("VALIDATING MARKET DATA")
    print("=" * 70)
    
    market_dir = PROJECT_ROOT / "data" / "raw" / "market"
    
    if not market_dir.exists():
        print("❌ Market data directory not found!")
        return False
    
    sectors = ["it", "financials", "pharma", "fmcg", "auto", "energy", 
               "metals", "realty", "telecom", "media", "infra"]
    
    total_rows = 0
    total_stocks = 0
    issues = []
    
    for sector in sectors:
        file_path = market_dir / f"{sector}_ohlcv.parquet"
        
        if not file_path.exists():
            print(f"⚠️  {sector.upper()}: File not found")
            continue
        
        try:
            df = pd.read_parquet(file_path)
            
            # Check required columns
            required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                issues.append(f"{sector}: Missing columns {missing_cols}")
                print(f"❌ {sector.upper()}: Missing columns {missing_cols}")
                continue
            
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Check for nulls
            null_counts = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum()
            if null_counts.sum() > 0:
                issues.append(f"{sector}: Has {null_counts.sum()} null values")
            
            # Check price logic (high >= low, etc.)
            invalid_prices = (df['high'] < df['low']).sum()
            if invalid_prices > 0:
                issues.append(f"{sector}: {invalid_prices} rows with high < low")
            
            # Check for negative prices
            negative_prices = ((df['open'] < 0) | (df['high'] < 0) | 
                             (df['low'] < 0) | (df['close'] < 0)).sum()
            if negative_prices > 0:
                issues.append(f"{sector}: {negative_prices} rows with negative prices")
            
            # Stats
            n_stocks = df['symbol'].nunique()
            n_rows = len(df)
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            
            total_rows += n_rows
            total_stocks += n_stocks
            
            print(f"✅ {sector.upper():12s}: {n_stocks:3d} stocks, {n_rows:6d} rows, {date_range}")
            
        except Exception as e:
            issues.append(f"{sector}: Error reading file - {e}")
            print(f"❌ {sector.upper()}: Error - {e}")
    
    # Check index data
    print("\nIndex Data:")
    index_file = market_dir / "NSEI_index.parquet"
    if index_file.exists():
        idx_df = pd.read_parquet(index_file)
        print(f"✅ NSEI Index: {len(idx_df)} rows, {idx_df['date'].min()} to {idx_df['date'].max()}")
    else:
        print("⚠️  NSEI Index: Not found")
    
    print(f"\n📊 Total: {total_stocks} stocks, {total_rows:,} rows")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ All market data looks good!")
        return True


def validate_macro_data():
    """Validate macro indicators."""
    print("\n" + "=" * 70)
    print("VALIDATING MACRO DATA")
    print("=" * 70)
    
    macro_file = PROJECT_ROOT / "data" / "raw" / "macro" / "macro_indicators.parquet"
    
    if not macro_file.exists():
        print("❌ Macro data file not found!")
        return False
    
    try:
        df = pd.read_parquet(macro_file)
        
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\n⚠️  Null values found:")
            print(null_counts[null_counts > 0])
        
        # Show sample statistics
        print(f"\nSample statistics:")
        print(df.describe().round(2))
        
        print("\n✅ Macro data looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading macro data: {e}")
        return False


def validate_fundamental_data():
    """Validate fundamental data."""
    print("\n" + "=" * 70)
    print("VALIDATING FUNDAMENTAL DATA")
    print("=" * 70)
    
    fund_file = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "fundamentals.parquet"
    
    if not fund_file.exists():
        print("❌ Fundamental data file not found!")
        return False
    
    try:
        df = pd.read_parquet(fund_file)
        
        print(f"Shape: {df.shape}")
        print(f"Stocks: {df['symbol'].nunique()}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\n⚠️  Null values found:")
            print(null_counts[null_counts > 0])
        
        # Show sample
        print(f"\nSample data:")
        print(df.head())
        
        print("\n✅ Fundamental data looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading fundamental data: {e}")
        return False


def validate_news_data():
    """Validate news data."""
    print("\n" + "=" * 70)
    print("VALIDATING NEWS DATA")
    print("=" * 70)
    
    news_file = PROJECT_ROOT / "data" / "raw" / "news" / "news_articles.parquet"
    
    if not news_file.exists():
        print("❌ News data file not found!")
        return False
    
    try:
        df = pd.read_parquet(news_file)
        
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Check sentiment range
        if 'sentiment' in df.columns:
            print(f"\nSentiment range: [{df['sentiment'].min():.2f}, {df['sentiment'].max():.2f}]")
            print(f"Sentiment mean: {df['sentiment'].mean():.2f}")
        
        # Show sample
        print(f"\nSample articles:")
        print(df[['date', 'title', 'sentiment']].head())
        
        print("\n✅ News data looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading news data: {e}")
        return False


def check_data_alignment():
    """Check if all data sources cover the same time period."""
    print("\n" + "=" * 70)
    print("CHECKING DATA ALIGNMENT")
    print("=" * 70)
    
    dates = {}
    
    # Market data
    market_file = PROJECT_ROOT / "data" / "raw" / "market" / "it_ohlcv.parquet"
    if market_file.exists():
        df = pd.read_parquet(market_file)
        dates['market'] = (df['date'].min(), df['date'].max())
    
    # Macro data
    macro_file = PROJECT_ROOT / "data" / "raw" / "macro" / "macro_indicators.parquet"
    if macro_file.exists():
        df = pd.read_parquet(macro_file)
        dates['macro'] = (df['date'].min(), df['date'].max())
    
    # News data
    news_file = PROJECT_ROOT / "data" / "raw" / "news" / "news_articles.parquet"
    if news_file.exists():
        df = pd.read_parquet(news_file)
        dates['news'] = (df['date'].min(), df['date'].max())
    
    print("Date ranges:")
    for source, (start, end) in dates.items():
        print(f"  {source:10s}: {start} to {end}")
    
    # Check overlap
    if len(dates) > 1:
        all_starts = [start for start, _ in dates.values()]
        all_ends = [end for _, end in dates.values()]
        
        overlap_start = max(all_starts)
        overlap_end = min(all_ends)
        
        print(f"\n📅 Common date range: {overlap_start} to {overlap_end}")
        
        if overlap_start <= overlap_end:
            print("✅ All data sources have overlapping dates!")
            return True
        else:
            print("❌ No overlapping dates found!")
            return False
    
    return True


def main():
    print("\n" + "=" * 70)
    print("HRL-SARP DATA VALIDATION")
    print("=" * 70)
    
    results = {
        'market': validate_market_data(),
        'macro': validate_macro_data(),
        'fundamental': validate_fundamental_data(),
        'news': validate_news_data(),
        'alignment': check_data_alignment(),
    }
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component.upper():15s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Data is ready for training!")
        print("\nNext step: python main.py train")
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED")
        print("Please fix the issues before training")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

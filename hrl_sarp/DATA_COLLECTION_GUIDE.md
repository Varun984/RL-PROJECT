# Data Collection Guide for HRL-SARP

## Quick Start: Generate Demo Data

The easiest way to get started is to generate synthetic demo data:

```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Generate demo data (2020-2023)
python main.py generate-demo

# Or specify custom date range
python main.py generate-demo --start 2018-01-01 --end 2023-12-31
```

This will create:
- `data/raw/market/` - OHLCV data for 55 stocks across 11 sectors
- `data/raw/macro/` - 12 macroeconomic indicators
- `data/raw/fundamentals/` - Fundamental ratios for all stocks
- `data/raw/news/` - Synthetic news articles with sentiment

---

## Real Data Collection (Requires API Access)

### Prerequisites

1. **yfinance** (Free, no API key needed)
   - Already installed
   - Works for NSE stocks with `.NS` suffix
   - Example: `TCS.NS`, `INFY.NS`

2. **Kite Connect** (Optional, for live data)
   - Sign up at: https://kite.zerodha.com/
   - Get API key and access token
   - Add to `.env` file:
     ```
     KITE_API_KEY=your_api_key
     KITE_ACCESS_TOKEN=your_access_token
     ```

3. **jugaad-data** (Optional, for NSE data)
   - Free, no API key needed
   - Already in requirements (commented out due to conflicts)

### Collect Real Data

```bash
# Collect all data types
python main.py collect-data --start 2020-01-01 --end 2023-12-31

# Skip specific data types
python main.py collect-data \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --skip-news \
    --skip-fundamentals

# Collect only market data
python main.py collect-data \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --skip-macro \
    --skip-fundamentals \
    --skip-news
```

---

## Data Structure

After collection, your data directory will look like:

```
data/
└── raw/
    ├── market/
    │   ├── it_ohlcv.parquet          # IT sector stocks
    │   ├── financials_ohlcv.parquet  # Financial sector stocks
    │   ├── pharma_ohlcv.parquet      # Pharma sector stocks
    │   ├── ... (11 sectors total)
    │   └── NSEI_index.parquet        # Nifty 50 index
    │
    ├── macro/
    │   └── macro_indicators.parquet  # GDP, inflation, rates, etc.
    │
    ├── fundamentals/
    │   └── fundamentals.parquet      # P/E, P/B, ROE, etc.
    │
    └── news/
        └── news_articles.parquet     # News with sentiment
```

---

## Stock Universe

### 11 Indian Sectors (55 stocks total)

1. **IT** (5 stocks)
   - TCS.NS, INFY.NS, WIPRO.NS, HCLTECH.NS, TECHM.NS

2. **Financials** (5 stocks)
   - HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS, SBIN.NS, AXISBANK.NS

3. **Pharma** (5 stocks)
   - SUNPHARMA.NS, DRREDDY.NS, CIPLA.NS, LUPIN.NS, DIVISLAB.NS

4. **FMCG** (5 stocks)
   - HINDUNILVR.NS, ITC.NS, NESTLEIND.NS, BRITANNIA.NS, DABUR.NS

5. **Auto** (5 stocks)
   - MARUTI.NS, TATAMOTORS.NS, M&M.NS, BAJAJ-AUTO.NS, EICHERMOT.NS

6. **Energy** (5 stocks)
   - RELIANCE.NS, ONGC.NS, NTPC.NS, POWERGRID.NS, COALINDIA.NS

7. **Metals** (5 stocks)
   - TATASTEEL.NS, HINDALCO.NS, JSWSTEEL.NS, VEDL.NS, NMDC.NS

8. **Realty** (5 stocks)
   - DLF.NS, GODREJPROP.NS, OBEROIRLTY.NS, PRESTIGE.NS, BRIGADE.NS

9. **Telecom** (3 stocks)
   - BHARTIARTL.NS, IDEA.NS, TATACOMM.NS

10. **Media** (4 stocks)
    - ZEEL.NS, SUNTV.NS, PVR.NS, DISHTV.NS

11. **Infra** (5 stocks)
    - LT.NS, ADANIPORTS.NS, IRBINFRA.NS, GMRINFRA.NS, NBCC.NS

---

## Macro Indicators Collected

1. **GDP Growth Rate** - Quarterly GDP growth (%)
2. **CPI Inflation** - Consumer Price Index inflation (%)
3. **Repo Rate** - RBI policy interest rate (%)
4. **IIP** - Industrial Production Index growth (%)
5. **PMI Manufacturing** - Manufacturing PMI (>50 = expansion)
6. **PMI Services** - Services PMI (>50 = expansion)
7. **FII Flows** - Foreign Institutional Investor net flows (₹ Cr)
8. **DII Flows** - Domestic Institutional Investor net flows (₹ Cr)
9. **Crude Oil** - Brent crude oil price (USD/barrel)
10. **USD/INR** - Exchange rate
11. **Gold** - Gold price (₹/10g)
12. **India VIX** - Volatility index

---

## Fundamental Metrics Collected

For each stock:
- **P/E Ratio** - Price to Earnings
- **P/B Ratio** - Price to Book
- **ROE** - Return on Equity (%)
- **Debt/Equity** - Debt to Equity ratio
- **Current Ratio** - Current assets / Current liabilities
- **Market Cap** - Market capitalization (₹ Cr)
- **Dividend Yield** - Annual dividend yield (%)
- **EPS** - Earnings Per Share
- **Revenue Growth** - YoY revenue growth (%)
- **Profit Margin** - Net profit margin (%)

---

## Data Sources

### Free Sources (No API Key)
1. **yfinance** - Yahoo Finance
   - OHLCV data for NSE stocks
   - Index data
   - Free, no rate limits

2. **RBI Website** - Reserve Bank of India
   - Macro indicators (GDP, inflation, rates)
   - Free, official data

3. **NSE India** - National Stock Exchange
   - Index data
   - Delivery data
   - Free, but rate limited

### Paid/API Key Required
1. **Kite Connect** (Zerodha)
   - Real-time data
   - Historical data
   - ₹2,000/month

2. **Screener.in** (Optional)
   - Fundamental data
   - Free tier available
   - Rate limited

---

## Troubleshooting

### Issue: "No module named 'numpy'"
**Solution**: Activate virtual environment first
```bash
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Issue: "Failed to fetch data for symbol"
**Solution**: 
- Check if symbol exists on NSE (use `.NS` suffix)
- Check internet connection
- Try with smaller date range
- Some stocks may be delisted

### Issue: "Rate limit exceeded"
**Solution**:
- Add delays between requests
- Use batch fetching
- Consider paid API (Kite Connect)

### Issue: "Macro data not available"
**Solution**:
- Macro data is monthly/quarterly
- May not be available for recent dates
- Use demo data for testing

---

## Next Steps After Data Collection

1. **Feature Engineering**
   ```bash
   python scripts/engineer_features.py
   ```
   Creates technical indicators, fundamental features, etc.

2. **Data Validation**
   ```bash
   python scripts/validate_data.py
   ```
   Checks for missing values, outliers, etc.

3. **Start Training**
   ```bash
   python main.py train
   ```
   Begins the 5-phase training pipeline

---

## Tips for Production

1. **Incremental Updates**
   - Run daily to collect latest data
   - Only fetch new dates, not entire history

2. **Data Quality**
   - Check for missing values
   - Handle corporate actions (splits, bonuses)
   - Adjust for dividends

3. **Storage**
   - Use Parquet format (compressed, fast)
   - Consider database for large datasets
   - Backup regularly

4. **Monitoring**
   - Log all data collection runs
   - Alert on failures
   - Track data freshness

---

## Example: Complete Workflow

```bash
# 1. Activate environment
.\venv\Scripts\activate

# 2. Generate demo data for testing
python main.py generate-demo --start 2020-01-01 --end 2023-12-31

# 3. Verify data was created
ls data/raw/market/
ls data/raw/macro/
ls data/raw/fundamentals/
ls data/raw/news/

# 4. (Optional) Collect real data
python main.py collect-data --start 2023-01-01 --end 2023-12-31

# 5. Engineer features
python scripts/engineer_features.py

# 6. Start training
python main.py train
```

---

## Questions?

**Q: How much data do I need?**
A: Minimum 3-5 years of daily data. More is better (10+ years ideal).

**Q: Can I use data from other sources?**
A: Yes! Just format it to match the expected structure (Parquet files with date, symbol, OHLCV columns).

**Q: How long does data collection take?**
A: Demo data: ~30 seconds. Real data: 10-30 minutes depending on date range and API limits.

**Q: Do I need all data types?**
A: Market data is essential. Macro, fundamentals, and news improve performance but are optional for testing.

**Q: Can I add more stocks?**
A: Yes! Edit `SECTOR_UNIVERSE` in `data_config.yaml` or the data fetcher scripts.

---

**Happy Data Collecting! 📊**

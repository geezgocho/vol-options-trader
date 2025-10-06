# Volatility Surface Analyzer - SPY & VIX

Complete options analysis system with 3D volatility surface modeling, VIX/SPY correlation analysis, and comprehensive Greeks calculations.

<img width="1114" height="813" alt="geezgocho-vol-options-trader-spy-iv-surface" src="https://github.com/user-attachments/assets/a4e45052-1375-408c-b92b-0acc65c6e873" />
<img width="1246" height="355" alt="geezgocho-vol-options-trader-spy-max-pain" src="https://github.com/user-attachments/assets/7e301601-15ca-44e8-9828-e2adafbdd7fa" />
<img width="1242" height="526" alt="geezgocho-vol-options-trader-spy-vix-corr" src="https://github.com/user-attachments/assets/fb1d625f-8387-432e-a194-a8c7ba932607" />
<img width="1250" height="476" alt="geezgocho-vol-options-trader-spy-skew-surface" src="https://github.com/user-attachments/assets/14ffbc07-09a3-4ff2-a1c5-ad6e7d5d4d5b" />

## 🚀 Quick Start (New to This?)

### **Fastest Way to Get Started**

```bash
# 1. Navigate to the project
cd finance/vol-options-trader

# 2. Install dependencies (one-time)
pip3 install -r requirements.txt

# 3. Run the analysis
python3 options_analysis_main.py

# 4. View the 3D surface (macOS)
open spy_vol_surface_3d.html
```

**That's it!** You'll get a full SPY volatility analysis in ~15 seconds. See [Getting Started Guide](#getting-started-for-beginners) below for more details.

---

## Features

### 📊 Options Chain Analysis
- Real-time options data fetching via yfinance
- Open Interest and Volume analysis
- Max Pain calculation
- Gamma Exposure analysis
- Put/Call ratios
- Support/Resistance levels from OI

### 🌊 Volatility Surface Modeling
- **3D Surface Interpolation**: Build complete volatility surfaces across strikes and expiries
- **Moneyness-based Analysis**: Normalized strikes for better surface fitting
- **Cubic Interpolation**: Smooth surface generation with fallback to linear/nearest neighbor
- **ATM Volatility Term Structure**: Track ATM IV across expiry dates

### 📈 Volatility Metrics
- **Skew Analysis**: Put IV vs Call IV across expiries
- **Term Structure**: ATM IV evolution over time (Contango/Backwardation)
- **Volatility Risk Premium**: VIX vs Realized Volatility spread

### ⚡ VIX/SPY Correlation
- Historical correlation analysis (30-day default)
- Realized volatility calculation
- Volatility risk premium signals
- Trend analysis with regression

### 📊 Interactive Visualizations
All plots are saved as interactive HTML files:
1. **Options Chain** - OI, Volume, and IV by strike
2. **Max Pain** - Pain distribution across strikes
3. **Gamma Exposure** - Net dealer gamma positioning
4. **3D Volatility Surface** - Full IV surface visualization
5. **Volatility Skew** - Put/Call IV spread analysis
6. **Term Structure** - ATM IV across expiries
7. **VIX/SPY Correlation** - Returns correlation and vol premium

## Installation

```bash
# Install dependencies (Python 3.11+ required)
pip3 install -r requirements.txt

# Or install individually
pip3 install pandas numpy scipy yfinance plotly
```

**Dependencies:**
- pandas ≥2.0.0 - Data manipulation
- numpy ≥1.24.0 - Numerical operations
- scipy ≥1.10.0 - Surface interpolation
- yfinance ≥0.2.28 - Options data fetching
- plotly ≥5.14.0 - Interactive visualizations

---

## Getting Started (For Beginners)

### **Option 1: Full Analysis (Recommended First Time)**

Run the complete analysis pipeline:

```bash
python3 options_analysis_main.py
```

**What happens:**
- ✅ Fetches SPY options data (5 expiries)
- ✅ Builds 3D volatility surface with cubic interpolation
- ✅ Analyzes VIX/SPY correlation (30 days)
- ✅ Calculates Max Pain, Gamma Exposure, Put/Call ratios
- ✅ Generates comprehensive summary in terminal
- ✅ Saves 6 interactive HTML visualizations

**Time:** ~10-15 seconds
**Output:** Console summary + 6 HTML files in current directory

---

### **Option 2: Interactive Python Session**

Experiment in Python step-by-step:

```bash
python3
```

Then paste:

```python
from options_analysis_main import OptionsAnalyzer, VolatilitySurface, OptionsVisualizer

# Quick SPY analysis
analyzer = OptionsAnalyzer()
results = analyzer.analyze_symbol('SPY', num_expiries=3)

# Print key metrics
print(f"SPY Spot: ${results['spot_price']:.2f}")
print(f"Max Pain: ${results['max_pain_strike']:.2f}")
print(f"Avg IV: {results['summary_stats']['avg_implied_volatility']*100:.1f}%")

# Build volatility surface
vol_surface = VolatilitySurface()
surface = vol_surface.build_surface(results['options_data'], results['spot_price'])

# Show 3D surface in browser (interactive!)
visualizer = OptionsVisualizer()
fig = visualizer.plot_volatility_surface_3d(surface, 'SPY')
fig.show()  # Opens in your default browser
```

---

### **Option 3: Example Scripts (Pre-built Modes)**

Run different analysis modes:

```bash
python3 example_usage.py
```

**Available modes** (edit file to uncomment):
- ✅ **Quick Analysis** - Fast SPY summary (runs by default)
- **Volatility Surface Only** - Focus on 3D surface + skew
- **VIX/SPY Correlation Only** - Vol premium analysis
- **Multi-Symbol Comparison** - SPY vs QQQ vs IWM
- **Custom Analysis** - Any ticker (AAPL, TSLA, NVDA, etc.)

---

## Usage

### Basic Usage
```bash
python3 options_analysis_main.py
```

This will:
1. Fetch SPY options data (5 expiries)
2. Build volatility surface with cubic interpolation
3. Analyze VIX/SPY correlation (30 days)
4. Generate comprehensive summary statistics
5. Save 6 interactive HTML visualizations

### Programmatic Usage
```python
from options_analysis_main import (
    OptionsAnalyzer,
    VolatilitySurface,
    VIXSPYAnalyzer,
    OptionsVisualizer
)

# Initialize components
analyzer = OptionsAnalyzer()
vol_surface = VolatilitySurface()
vix_spy = VIXSPYAnalyzer()

# Analyze SPY options
spy_results = analyzer.analyze_symbol('SPY', num_expiries=5)

# Build volatility surface
surface_data = vol_surface.build_surface(
    spy_results['options_data'],
    spy_results['spot_price']
)

# Get VIX/SPY metrics
vix_spy_data = vix_spy.fetch_historical_data(days_back=30)
vol_risk_premium = vix_spy.calculate_volatility_risk_premium()

# Create visualizations
visualizer = OptionsVisualizer()
surface_fig = visualizer.plot_volatility_surface_3d(surface_data, 'SPY')
surface_fig.show()
```

## Output

### Console Output
```
================================================================================
VOLATILITY SURFACE ANALYZER - SPY & VIX
================================================================================

[1/4] Fetching SPY options data...
[2/4] Building volatility surface...
[3/4] Analyzing VIX/SPY correlation...
[4/4] Analysis complete!

================================================================================
SPY OPTIONS SUMMARY
================================================================================

📊 Market Data:
   Spot Price: $XXX.XX
   Max Pain: $XXX.XX
   Put/Call Volume Ratio: X.XX
   Put/Call OI Ratio: X.XX

📈 Key Levels:
   Support: [XXX.X, XXX.X, XXX.X]
   Resistance: [XXX.X, XXX.X, XXX.X]

📋 Trading Activity:
   Total Volume: XXX,XXX
   Total Open Interest: X,XXX,XXX
   Average IV: XX.X%
   Number of Strikes: XXX
   Number of Expiries: X

🌊 Volatility Surface:
   Near-term ATM IV: XX.XX%
   Far-term ATM IV: XX.XX%
   Term Structure: Contango/Backwardation
   Average Skew: X.XX% (Put IV - Call IV)

⚡ VIX/SPY Analysis:
   Current VIX: XX.XX
   SPY Realized Vol (30d): XX.XX%
   Volatility Risk Premium: X.XX%
   SPY-VIX Correlation: -0.XXX
   💡 Signal: [Trading signal based on vol premium]
```

### HTML Files Generated
All plots saved to `vol-options-trader/` directory:
- `spy_options_chain.html`
- `spy_max_pain.html`
- `spy_gamma_exposure.html`
- `spy_vol_surface_3d.html` ⭐
- `spy_vol_skew.html`
- `spy_term_structure.html`
- `vix_spy_correlation.html`

## Architecture

### Core Classes

**OptionsDataFetcher**
- Fetches options chains from yfinance
- Supports multiple expiries
- Handles spot price retrieval

**GreeksCalculator**
- Moneyness classification (ITM/ATM/OTM)
- Max Pain calculation
- Gamma exposure by strike

**VolatilitySurface** ⭐
- 3D surface interpolation (cubic/linear/nearest)
- ATM volatility extraction
- Skew calculation (OTM puts vs calls)
- Term structure analysis

**VIXSPYAnalyzer** ⭐
- Historical VIX/SPY data fetching
- Correlation calculation
- Realized volatility computation
- Volatility risk premium

**OptionsAnalyzer**
- Main analysis orchestration
- Put/Call ratio calculations
- Key level identification
- Summary statistics

**OptionsVisualizer**
- All plotting methods
- Interactive Plotly charts
- 3D surface visualization

## Trading Signals

### Volatility Risk Premium
- **> 5%**: High premium - consider selling volatility (credit spreads, covered calls)
- **< -5%**: Negative premium - consider buying volatility (long options, protective puts)
- **-5% to 5%**: Neutral - no clear vol signal

### Volatility Skew
- **Positive Skew**: Puts more expensive than calls (typical for SPY) - market fears downside
- **Negative Skew**: Calls more expensive - bullish positioning or takeover speculation

### Term Structure
- **Contango** (far > near): Normal market - selling near-term premium attractive
- **Backwardation** (near > far): Stress/event - short-term uncertainty priced in

## Data Sources

- **Options Data**: yfinance (free, 15-min delayed)
- **VIX Data**: Yahoo Finance ^VIX ticker
- **SPY Data**: Yahoo Finance SPY ticker

## Customization

### Change Symbols
```python
# Analyze any optionable stock
results = analyzer.analyze_symbol('AAPL', num_expiries=5)

# Build surface for different ticker
surface_data = vol_surface.build_surface(
    results['options_data'],
    results['spot_price']
)
```

### Adjust Time Horizons
```python
# Longer historical analysis
vix_spy_data = vix_spy.fetch_historical_data(days_back=90)

# More expiries for better surface
spy_results = analyzer.analyze_symbol('SPY', num_expiries=10)
```

### Analyze Any Optionable Stock

Quick custom analysis for any ticker:

```python
from example_usage import custom_analysis

# Analyze AAPL with 4 expiries
results, surface = custom_analysis('AAPL', num_expiries=4)

# Or any other stock
results, surface = custom_analysis('TSLA', num_expiries=5)
results, surface = custom_analysis('NVDA', num_expiries=3)
```

Saves HTML files: `aapl_options_chain.html` and `aapl_vol_surface.html`

### Surface Parameters
Modify in `VolatilitySurface.build_surface()`:
```python
# Wider moneyness range
moneyness_range = np.linspace(0.7, 1.3, 50)  # Default: 0.8-1.2

# Higher resolution grid
expiry_range = np.linspace(min_dte, max_dte, 50)  # Default: 30
```

## Understanding the Output

### **Key Metrics Explained**

**Volatility Risk Premium = VIX - Realized Vol**
- `> 5%` → VIX expensive - Sell volatility (credit spreads, covered calls)
- `< -5%` → VIX cheap - Buy volatility (long options, straddles)
- `-5% to 5%` → Neutral - no clear signal

**Volatility Skew = Put IV - Call IV**
- `Positive` → Puts more expensive (normal - downside fear)
- `Higher skew` → More fear premium in puts
- Use for: Put selling strategies when skew is elevated

**Term Structure**
- `Contango` (far > near) → Normal market, sell front-month
- `Backwardation` (near > far) → Stress/event expected soon

**Put/Call OI Ratio**
- `> 1.5` → Bearish positioning (heavy put buying)
- `< 0.7` → Bullish positioning (heavy call buying)
- `~1.0` → Neutral positioning

### **Example Live Output**

From a recent run:
```
SPY Spot: $669.21
Max Pain: $665.00
Put/Call OI Ratio: 2.66 (bearish)

Volatility Surface:
  Near-term ATM IV: 5.88%
  Far-term ATM IV: 10.29%
  Term Structure: Contango ✓
  Average Skew: 18.16% (high put premium)

VIX/SPY Analysis:
  Current VIX: 16.65
  Realized Vol: 5.87%
  Vol Risk Premium: 10.78% 🔥
  Signal: HIGH vol premium → SELL volatility
```

**Trading Implication:**
Vol premium of 10.78% suggests selling volatility strategies (credit spreads, iron condors) are favorable. High skew (18.16%) indicates put selling is particularly attractive.

---

## Limitations

1. **Data Delay**: yfinance has ~15min delay (use paid API for real-time)
2. **VIX Options**: Not available via yfinance - using VIX index only
3. **Greeks**: Uses yfinance Greeks (limited accuracy vs. professional tools)
4. **Surface Fitting**: Edge cases may have interpolation artifacts
5. **After Hours**: Data quality varies outside market hours (9:30-16:00 ET best)

## Troubleshooting

### **"ModuleNotFoundError: No module named 'pandas'"**
```bash
pip3 install -r requirements.txt
```

### **"command not found: python"**
Use `python3` instead of `python`:
```bash
python3 options_analysis_main.py
```

### **Plots not opening in browser**
Manually open the HTML files:
```bash
# macOS
open spy_vol_surface_3d.html

# Linux
xdg-open spy_vol_surface_3d.html

# Windows
start spy_vol_surface_3d.html
```

### **Empty/Error on VIX data**
VIX historical data may have gaps. Adjust time horizon:
```python
vix_spy_data = vix_spy.fetch_historical_data(days_back=60)  # Try longer period
```

## Future Enhancements

- [ ] Add historical options data persistence (SQLite/PostgreSQL)
- [ ] Implement Black-Scholes Greeks calculation
- [ ] VIX futures term structure (requires CBOE data)
- [ ] IV rank and IV percentile calculations
- [ ] Alerts for vol premium thresholds
- [ ] Multi-symbol comparison dashboard
- [ ] Integration with TD Ameritrade API (real-time)
- [ ] Machine learning vol surface fitting

## License

MIT

## Author

Built for quantitative finance analysis and volatility trading research.

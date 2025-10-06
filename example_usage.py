"""
Example usage patterns for the Volatility Surface Analyzer
"""

from options_analysis_main import (
    OptionsAnalyzer,
    VolatilitySurface,
    VIXSPYAnalyzer,
    OptionsVisualizer
)


def quick_analysis():
    """Quick SPY analysis with default settings"""
    print("=" * 60)
    print("QUICK ANALYSIS MODE")
    print("=" * 60)

    analyzer = OptionsAnalyzer()
    results = analyzer.analyze_symbol('SPY', num_expiries=3)

    if 'summary_stats' in results:
        print(f"\nSPY Spot: ${results['spot_price']:.2f}")
        print(f"Max Pain: ${results['max_pain_strike']:.2f}")
        print(f"Put/Call Ratio: {results['pc_oi_ratio']:.2f}")
        print(f"Average IV: {results['summary_stats']['avg_implied_volatility']*100:.1f}%")

    return results


def vol_surface_only():
    """Focus on volatility surface analysis"""
    print("\n" + "=" * 60)
    print("VOLATILITY SURFACE MODE")
    print("=" * 60)

    analyzer = OptionsAnalyzer()
    vol_surface = VolatilitySurface()
    visualizer = OptionsVisualizer()

    # Get options data
    spy_results = analyzer.analyze_symbol('SPY', num_expiries=5)

    # Build surface
    surface_data = vol_surface.build_surface(
        spy_results['options_data'],
        spy_results['spot_price']
    )

    if 'error' not in surface_data:
        # Show term structure
        term_struct = surface_data['term_structure']
        print("\nVolatility Term Structure:")
        print(term_struct[['days_to_expiry', 'atm_iv']])

        # Show skew
        skew = surface_data['skew']
        print("\nVolatility Skew:")
        print(skew[['days_to_expiry', 'put_iv', 'call_iv', 'skew']])

        # Create 3D plot
        fig = visualizer.plot_volatility_surface_3d(surface_data, 'SPY')
        fig.show()

    return surface_data


def vix_spy_only():
    """Focus on VIX/SPY correlation"""
    print("\n" + "=" * 60)
    print("VIX/SPY CORRELATION MODE")
    print("=" * 60)

    vix_spy = VIXSPYAnalyzer()
    visualizer = OptionsVisualizer()

    # Fetch 60 days of data for better correlation
    data = vix_spy.fetch_historical_data(days_back=60)

    print(f"\nVIX Level: {data['current_vix']:.2f}")
    print(f"SPY Realized Vol: {data['realized_vol_spy']:.2f}%")
    print(f"Vol Risk Premium: {data['vix_spread']:.2f}%")
    print(f"Correlation: {data['spy_vix_correlation']:.3f}")

    # Plot
    fig = visualizer.plot_vix_spy_correlation(data)
    fig.show()

    return data


def multi_symbol_comparison():
    """Compare volatility across multiple symbols"""
    print("\n" + "=" * 60)
    print("MULTI-SYMBOL COMPARISON")
    print("=" * 60)

    analyzer = OptionsAnalyzer()
    symbols = ['SPY', 'QQQ', 'IWM']  # SPY, Nasdaq, Russell 2000

    results = {}
    for symbol in symbols:
        try:
            print(f"\nAnalyzing {symbol}...")
            res = analyzer.analyze_symbol(symbol, num_expiries=3)
            results[symbol] = res

            if 'summary_stats' in res:
                print(f"  Spot: ${res['spot_price']:.2f}")
                print(f"  Avg IV: {res['summary_stats']['avg_implied_volatility']*100:.1f}%")
                print(f"  P/C Ratio: {res['pc_oi_ratio']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")

    return results


def custom_analysis(symbol: str, num_expiries: int = 5, days_back: int = 30):
    """
    Fully customizable analysis

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'TSLA')
        num_expiries: Number of expiry dates to analyze
        days_back: Historical days for VIX/SPY correlation
    """
    print("\n" + "=" * 60)
    print(f"CUSTOM ANALYSIS - {symbol}")
    print("=" * 60)

    analyzer = OptionsAnalyzer()
    vol_surface = VolatilitySurface()
    visualizer = OptionsVisualizer()

    # Options analysis
    results = analyzer.analyze_symbol(symbol, num_expiries=num_expiries)

    # Build surface
    surface_data = vol_surface.build_surface(
        results['options_data'],
        results['spot_price']
    )

    # Summary
    if 'summary_stats' in results:
        print(f"\n{symbol} Analysis:")
        print(f"  Spot Price: ${results['spot_price']:.2f}")
        print(f"  Max Pain: ${results['max_pain_strike']:.2f}")
        print(f"  Avg IV: {results['summary_stats']['avg_implied_volatility']*100:.1f}%")
        print(f"  Total OI: {results['summary_stats']['total_open_interest']:,.0f}")

    # Create visualizations
    chain_fig = visualizer.plot_options_chain(results)
    chain_fig.write_html(f'{symbol.lower()}_options_chain.html')

    if 'error' not in surface_data:
        surface_fig = visualizer.plot_volatility_surface_3d(surface_data, symbol)
        surface_fig.write_html(f'{symbol.lower()}_vol_surface.html')

    print(f"\n✅ Saved plots to {symbol.lower()}_*.html")

    return results, surface_data


if __name__ == "__main__":
    # Run different analysis modes

    # Mode 1: Quick analysis
    quick_results = quick_analysis()

    # Mode 2: Volatility surface focus (uncomment to run)
    # surface_data = vol_surface_only()

    # Mode 3: VIX/SPY correlation (uncomment to run)
    # vix_data = vix_spy_only()

    # Mode 4: Multi-symbol comparison (uncomment to run)
    # multi_results = multi_symbol_comparison()

    # Mode 5: Custom analysis for any symbol (uncomment to run)
    # aapl_results, aapl_surface = custom_analysis('AAPL', num_expiries=4)

    print("\n" + "=" * 60)
    print("✨ Done! Uncomment other modes to try them.")
    print("=" * 60)

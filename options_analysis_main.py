"""
Options Chain Analysis System for SPY and VIX
Main application with data fetching, analysis, and visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata, RBFInterpolator
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class OptionsDataFetcher:
    """
    Fetches options data from various sources including yfinance and TD Ameritrade
    """
    
    def __init__(self, use_td_api: bool = False, td_config: Optional[Dict] = None):
        """
        Initialize the data fetcher
        
        Args:
            use_td_api: Whether to use TD Ameritrade API
            td_config: Configuration for TD API (client_id, refresh_token, etc.)
        """
        self.use_td_api = use_td_api
        self.td_config = td_config
        
        if use_td_api and td_config:
            self._setup_td_api()
    
    def _setup_td_api(self):
        """Setup TD Ameritrade API connection"""
        # TD Ameritrade API setup would go here
        # This requires authentication tokens
        pass
    
    def fetch_options_chain_yfinance(self, symbol: str, expiry_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch options chain using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'VIX')
            expiry_date: Optional expiry date in YYYY-MM-DD format
        
        Returns:
            DataFrame with options chain data
        """
        ticker = yf.Ticker(symbol)
        
        # Get available expiry dates
        expiry_dates = ticker.options
        
        if not expiry_dates:
            raise ValueError(f"No options data available for {symbol}")
        
        # Use provided expiry or get the nearest one
        if expiry_date and expiry_date in expiry_dates:
            target_expiry = expiry_date
        else:
            target_expiry = expiry_dates[0]  # Use nearest expiry
        
        # Get options chain
        opt_chain = ticker.option_chain(target_expiry)
        
        # Process calls
        calls = opt_chain.calls.copy()
        calls['type'] = 'CALL'
        calls['expiry'] = target_expiry
        
        # Process puts
        puts = opt_chain.puts.copy()
        puts['type'] = 'PUT'
        puts['expiry'] = target_expiry
        
        # Combine and return
        options_df = pd.concat([calls, puts], ignore_index=True)
        options_df['symbol'] = symbol
        options_df['fetch_time'] = datetime.now()
        
        return options_df
    
    def fetch_multiple_expiries(self, symbol: str, num_expiries: int = 5) -> pd.DataFrame:
        """
        Fetch options chains for multiple expiry dates
        
        Args:
            symbol: Stock symbol
            num_expiries: Number of expiry dates to fetch
        
        Returns:
            Combined DataFrame with all options data
        """
        ticker = yf.Ticker(symbol)
        expiry_dates = ticker.options[:num_expiries]
        
        all_options = []
        for expiry in expiry_dates:
            try:
                options_df = self.fetch_options_chain_yfinance(symbol, expiry)
                all_options.append(options_df)
            except Exception as e:
                print(f"Error fetching {symbol} options for {expiry}: {e}")
        
        if all_options:
            return pd.concat(all_options, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_spot_price(self, symbol: str) -> float:
        """Get current spot price for the underlying"""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return 0.0


class GreeksCalculator:
    """
    Calculate and analyze options Greeks
    """
    
    @staticmethod
    def calculate_moneyness(spot: float, strike: float, option_type: str) -> str:
        """
        Determine if option is ITM, ATM, or OTM
        
        Args:
            spot: Current price of underlying
            strike: Strike price
            option_type: 'CALL' or 'PUT'
        
        Returns:
            'ITM', 'ATM', or 'OTM'
        """
        threshold = 0.01  # 1% threshold for ATM
        
        if option_type == 'CALL':
            if spot > strike * (1 + threshold):
                return 'ITM'
            elif spot < strike * (1 - threshold):
                return 'OTM'
            else:
                return 'ATM'
        else:  # PUT
            if spot < strike * (1 - threshold):
                return 'ITM'
            elif spot > strike * (1 + threshold):
                return 'OTM'
            else:
                return 'ATM'
    
    @staticmethod
    def calculate_max_pain(options_df: pd.DataFrame, spot_price: float) -> Tuple[float, pd.DataFrame]:
        """
        Calculate max pain strike price
        
        Args:
            options_df: DataFrame with options data
            spot_price: Current spot price
        
        Returns:
            Tuple of (max_pain_strike, pain_by_strike DataFrame)
        """
        if options_df.empty:
            return 0.0, pd.DataFrame()
        
        strikes = options_df['strike'].unique()
        pain_by_strike = []
        
        for strike in strikes:
            call_pain = 0
            put_pain = 0
            
            # Calculate pain for calls
            calls = options_df[(options_df['type'] == 'CALL')]
            for _, call in calls.iterrows():
                if call['strike'] < strike:
                    call_pain += (strike - call['strike']) * call.get('openInterest', 0)
            
            # Calculate pain for puts
            puts = options_df[(options_df['type'] == 'PUT')]
            for _, put in puts.iterrows():
                if put['strike'] > strike:
                    put_pain += (put['strike'] - strike) * put.get('openInterest', 0)
            
            total_pain = call_pain + put_pain
            pain_by_strike.append({
                'strike': strike,
                'call_pain': call_pain,
                'put_pain': put_pain,
                'total_pain': total_pain
            })
        
        pain_df = pd.DataFrame(pain_by_strike)
        if not pain_df.empty:
            max_pain_strike = pain_df.loc[pain_df['total_pain'].idxmin(), 'strike']
            return max_pain_strike, pain_df
        
        return 0.0, pd.DataFrame()
    
    @staticmethod
    def calculate_gamma_exposure(options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """
        Calculate gamma exposure by strike
        
        Args:
            options_df: DataFrame with options data
            spot_price: Current spot price
        
        Returns:
            DataFrame with gamma exposure by strike
        """
        if options_df.empty or 'gamma' not in options_df.columns:
            return pd.DataFrame()
        
        # Group by strike and calculate net gamma
        gamma_exposure = options_df.groupby(['strike', 'type']).agg({
            'gamma': 'sum',
            'openInterest': 'sum',
            'volume': 'sum'
        }).reset_index()
        
        # Calculate dollar gamma
        gamma_exposure['dollar_gamma'] = gamma_exposure['gamma'] * gamma_exposure['openInterest'] * 100 * spot_price
        
        # Pivot to get calls and puts side by side
        gamma_pivot = gamma_exposure.pivot(index='strike', columns='type', values='dollar_gamma').fillna(0)
        gamma_pivot['net_gamma'] = gamma_pivot.get('CALL', 0) - gamma_pivot.get('PUT', 0)
        
        return gamma_pivot.reset_index()


class OptionsAnalyzer:
    """
    Main analyzer class for options chain analysis
    """
    
    def __init__(self):
        self.fetcher = OptionsDataFetcher()
        self.greeks_calc = GreeksCalculator()
        self.current_data = {}
    
    def analyze_symbol(self, symbol: str, num_expiries: int = 3) -> Dict:
        """
        Perform comprehensive analysis on a symbol's options
        
        Args:
            symbol: Stock symbol to analyze
            num_expiries: Number of expiry dates to analyze
        
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing {symbol} options...")
        
        # Fetch data
        options_df = self.fetcher.fetch_multiple_expiries(symbol, num_expiries)
        spot_price = self.fetcher.get_spot_price(symbol)
        
        if options_df.empty:
            return {'error': f'No options data available for {symbol}'}
        
        # Add moneyness
        options_df['moneyness'] = options_df.apply(
            lambda row: self.greeks_calc.calculate_moneyness(
                spot_price, row['strike'], row['type']
            ), axis=1
        )
        
        # Calculate metrics
        max_pain_strike, pain_df = self.greeks_calc.calculate_max_pain(options_df, spot_price)
        gamma_exposure = self.greeks_calc.calculate_gamma_exposure(options_df, spot_price)
        
        # Calculate put/call ratios
        pc_volume_ratio = self._calculate_pc_ratio(options_df, 'volume')
        pc_oi_ratio = self._calculate_pc_ratio(options_df, 'openInterest')
        
        # Find key levels
        key_levels = self._find_key_levels(options_df, spot_price)
        
        # Store results
        results = {
            'symbol': symbol,
            'spot_price': spot_price,
            'timestamp': datetime.now(),
            'options_data': options_df,
            'max_pain_strike': max_pain_strike,
            'pain_distribution': pain_df,
            'gamma_exposure': gamma_exposure,
            'pc_volume_ratio': pc_volume_ratio,
            'pc_oi_ratio': pc_oi_ratio,
            'key_levels': key_levels,
            'summary_stats': self._calculate_summary_stats(options_df)
        }
        
        self.current_data[symbol] = results
        return results
    
    def _calculate_pc_ratio(self, options_df: pd.DataFrame, column: str) -> float:
        """Calculate put/call ratio for a given column"""
        puts_sum = options_df[options_df['type'] == 'PUT'][column].sum()
        calls_sum = options_df[options_df['type'] == 'CALL'][column].sum()
        
        if calls_sum > 0:
            return puts_sum / calls_sum
        return 0.0
    
    def _find_key_levels(self, options_df: pd.DataFrame, spot_price: float) -> Dict:
        """Find key support and resistance levels based on open interest"""
        # Group by strike and sum open interest
        oi_by_strike = options_df.groupby('strike')['openInterest'].sum().sort_values(ascending=False)
        
        # Find top strikes with highest OI
        top_strikes = oi_by_strike.head(5).index.tolist()
        
        # Separate into support and resistance
        support_levels = [s for s in top_strikes if s < spot_price]
        resistance_levels = [s for s in top_strikes if s > spot_price]
        
        return {
            'support': sorted(support_levels, reverse=True)[:3],
            'resistance': sorted(resistance_levels)[:3],
            'high_oi_strikes': top_strikes
        }
    
    def _calculate_summary_stats(self, options_df: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        return {
            'total_volume': options_df['volume'].sum(),
            'total_open_interest': options_df['openInterest'].sum(),
            'avg_implied_volatility': options_df['impliedVolatility'].mean(),
            'num_strikes': options_df['strike'].nunique(),
            'num_expiries': options_df['expiry'].nunique(),
            'calls_count': len(options_df[options_df['type'] == 'CALL']),
            'puts_count': len(options_df[options_df['type'] == 'PUT'])
        }


class VolatilitySurface:
    """
    Model and analyze the volatility surface across strikes and expiries
    """

    def __init__(self):
        self.surface_data = None
        self.grid_strikes = None
        self.grid_expiries = None
        self.grid_iv = None

    def build_surface(self, options_df: pd.DataFrame, spot_price: float) -> Dict:
        """
        Build volatility surface from options data

        Args:
            options_df: DataFrame with options data including IV
            spot_price: Current underlying price

        Returns:
            Dictionary with surface data and metadata
        """
        if options_df.empty or 'impliedVolatility' not in options_df.columns:
            return {'error': 'Insufficient data for surface construction'}

        # Filter out zero IV and NaN values
        df = options_df[
            (options_df['impliedVolatility'] > 0) &
            (options_df['impliedVolatility'].notna())
        ].copy()

        if df.empty:
            return {'error': 'No valid IV data'}

        # Calculate moneyness (strike/spot) and days to expiration
        df['moneyness'] = df['strike'] / spot_price
        df['days_to_expiry'] = df['expiry'].apply(
            lambda x: (pd.to_datetime(x) - pd.Timestamp.now()).days
        )

        # Filter reasonable values
        df = df[
            (df['moneyness'] > 0.7) &
            (df['moneyness'] < 1.3) &
            (df['days_to_expiry'] > 0)
        ]

        if df.empty:
            return {'error': 'No data in reasonable moneyness range'}

        # Create grid for interpolation
        moneyness_range = np.linspace(0.8, 1.2, 50)
        expiry_range = np.linspace(
            df['days_to_expiry'].min(),
            df['days_to_expiry'].max(),
            30
        )

        self.grid_strikes = moneyness_range * spot_price
        self.grid_expiries = expiry_range

        # Prepare data for interpolation
        points = df[['moneyness', 'days_to_expiry']].values
        values = df['impliedVolatility'].values

        # Create meshgrid
        X, Y = np.meshgrid(moneyness_range, expiry_range)

        # Interpolate using griddata (linear or cubic)
        try:
            Z = griddata(points, values, (X, Y), method='cubic')
            # Fill NaN values with nearest neighbor
            Z_nearest = griddata(points, values, (X, Y), method='nearest')
            Z = np.where(np.isnan(Z), Z_nearest, Z)
        except Exception as e:
            print(f"Cubic interpolation failed: {e}, falling back to linear")
            Z = griddata(points, values, (X, Y), method='linear')
            Z_nearest = griddata(points, values, (X, Y), method='nearest')
            Z = np.where(np.isnan(Z), Z_nearest, Z)

        self.grid_iv = Z
        self.surface_data = df

        # Calculate surface metrics
        atm_iv = self._get_atm_volatility(spot_price)
        skew = self._calculate_skew(spot_price)
        term_structure = self._calculate_term_structure(spot_price)

        return {
            'grid_strikes': self.grid_strikes,
            'grid_expiries': self.grid_expiries,
            'grid_iv': self.grid_iv,
            'atm_iv': atm_iv,
            'skew': skew,
            'term_structure': term_structure,
            'raw_data': df
        }

    def _get_atm_volatility(self, spot_price: float) -> pd.DataFrame:
        """Get ATM implied volatility across expiries"""
        if self.surface_data is None:
            return pd.DataFrame()

        df = self.surface_data
        atm_ivs = []

        for expiry in df['expiry'].unique():
            expiry_data = df[df['expiry'] == expiry]
            # Find closest to ATM
            atm_row = expiry_data.iloc[(expiry_data['moneyness'] - 1.0).abs().argsort()[:1]]
            if not atm_row.empty:
                atm_ivs.append({
                    'expiry': expiry,
                    'days_to_expiry': atm_row['days_to_expiry'].iloc[0],
                    'atm_iv': atm_row['impliedVolatility'].iloc[0]
                })

        return pd.DataFrame(atm_ivs)

    def _calculate_skew(self, spot_price: float) -> pd.DataFrame:
        """Calculate volatility skew for each expiry"""
        if self.surface_data is None:
            return pd.DataFrame()

        df = self.surface_data
        skew_data = []

        for expiry in df['expiry'].unique():
            expiry_data = df[df['expiry'] == expiry]

            # Get OTM put and call IVs
            otm_puts = expiry_data[
                (expiry_data['type'] == 'PUT') &
                (expiry_data['moneyness'] < 0.95)
            ]
            otm_calls = expiry_data[
                (expiry_data['type'] == 'CALL') &
                (expiry_data['moneyness'] > 1.05)
            ]

            if not otm_puts.empty and not otm_calls.empty:
                put_iv = otm_puts['impliedVolatility'].mean()
                call_iv = otm_calls['impliedVolatility'].mean()
                skew = put_iv - call_iv

                skew_data.append({
                    'expiry': expiry,
                    'days_to_expiry': expiry_data['days_to_expiry'].iloc[0],
                    'put_iv': put_iv,
                    'call_iv': call_iv,
                    'skew': skew
                })

        return pd.DataFrame(skew_data)

    def _calculate_term_structure(self, spot_price: float) -> pd.DataFrame:
        """Calculate term structure of volatility"""
        atm_vols = self._get_atm_volatility(spot_price)
        if not atm_vols.empty:
            return atm_vols.sort_values('days_to_expiry')
        return pd.DataFrame()


class VIXSPYAnalyzer:
    """
    Analyze VIX and SPY correlation and volatility relationships
    """

    def __init__(self):
        self.vix_data = None
        self.spy_data = None
        self.correlation_data = None

    def fetch_historical_data(self, days_back: int = 30) -> Dict:
        """
        Fetch historical VIX and SPY data

        Args:
            days_back: Number of days of historical data

        Returns:
            Dictionary with VIX and SPY data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Fetch VIX
        vix = yf.Ticker('^VIX')
        vix_hist = vix.history(start=start_date, end=end_date)

        # Fetch SPY
        spy = yf.Ticker('SPY')
        spy_hist = spy.history(start=start_date, end=end_date)

        self.vix_data = vix_hist
        self.spy_data = spy_hist

        # Calculate returns
        spy_returns = spy_hist['Close'].pct_change()
        vix_changes = vix_hist['Close'].pct_change()

        # Calculate correlation
        correlation = spy_returns.corr(vix_changes)

        # Calculate realized volatility of SPY
        realized_vol = spy_returns.std() * np.sqrt(252) * 100

        # Compare with current VIX
        current_vix = vix_hist['Close'].iloc[-1]

        self.correlation_data = {
            'spy_vix_correlation': correlation,
            'realized_vol_spy': realized_vol,
            'current_vix': current_vix,
            'vix_spread': current_vix - realized_vol,
            'spy_returns': spy_returns,
            'vix_changes': vix_changes
        }

        return self.correlation_data

    def analyze_vix_term_structure(self) -> pd.DataFrame:
        """
        Analyze VIX futures term structure (if available)
        Note: This requires VIX futures data which may need additional data source
        """
        # Placeholder for VIX futures analysis
        # Would require additional data source like CBOE API
        return pd.DataFrame()

    def calculate_volatility_risk_premium(self) -> float:
        """
        Calculate the volatility risk premium (VIX - Realized Vol)
        """
        if self.correlation_data is None:
            self.fetch_historical_data()

        return self.correlation_data['vix_spread']


class OptionsVisualizer:
    """
    Create visualizations for options analysis
    """

    @staticmethod
    def plot_options_chain(analysis_results: Dict) -> go.Figure:
        """Create interactive options chain visualization"""
        if 'options_data' not in analysis_results:
            return go.Figure()
        
        df = analysis_results['options_data']
        spot_price = analysis_results['spot_price']
        
        # Filter for nearest expiry for cleaner visualization
        nearest_expiry = df['expiry'].min()
        df_filtered = df[df['expiry'] == nearest_expiry]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Open Interest by Strike', 'Volume by Strike', 'Implied Volatility'),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Open Interest plot
        calls = df_filtered[df_filtered['type'] == 'CALL']
        puts = df_filtered[df_filtered['type'] == 'PUT']
        
        fig.add_trace(
            go.Bar(x=calls['strike'], y=calls['openInterest'], name='Call OI', 
                   marker_color='green', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=puts['strike'], y=puts['openInterest'], name='Put OI',
                   marker_color='red', opacity=0.7),
            row=1, col=1
        )
        
        # Add spot price line
        fig.add_vline(x=spot_price, line_dash="dash", line_color="blue",
                      annotation_text=f"Spot: {spot_price:.2f}", row=1, col=1)
        
        # Volume plot
        fig.add_trace(
            go.Bar(x=calls['strike'], y=calls['volume'], name='Call Volume',
                   marker_color='lightgreen', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=puts['strike'], y=puts['volume'], name='Put Volume',
                   marker_color='lightcoral', showlegend=False),
            row=2, col=1
        )
        
        # Implied Volatility plot
        fig.add_trace(
            go.Scatter(x=calls['strike'], y=calls['impliedVolatility']*100,
                      mode='markers+lines', name='Call IV', marker_color='green'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=puts['strike'], y=puts['impliedVolatility']*100,
                      mode='markers+lines', name='Put IV', marker_color='red'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{analysis_results['symbol']} Options Chain Analysis - Expiry: {nearest_expiry}",
            height=900,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Strike Price", row=3, col=1)
        fig.update_yaxes(title_text="Open Interest", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="IV (%)", row=3, col=1)
        
        return fig
    
    @staticmethod
    def plot_max_pain(analysis_results: Dict) -> go.Figure:
        """Create max pain visualization"""
        if 'pain_distribution' not in analysis_results:
            return go.Figure()
        
        pain_df = analysis_results['pain_distribution']
        max_pain_strike = analysis_results['max_pain_strike']
        spot_price = analysis_results['spot_price']
        
        fig = go.Figure()
        
        # Plot pain distribution
        fig.add_trace(go.Bar(
            x=pain_df['strike'],
            y=pain_df['total_pain'],
            name='Total Pain',
            marker_color='purple',
            opacity=0.6
        ))
        
        # Add max pain line
        fig.add_vline(x=max_pain_strike, line_dash="dash", line_color="red",
                      annotation_text=f"Max Pain: {max_pain_strike:.2f}")
        
        # Add spot price line
        fig.add_vline(x=spot_price, line_dash="dash", line_color="blue",
                      annotation_text=f"Spot: {spot_price:.2f}")
        
        fig.update_layout(
            title=f"{analysis_results['symbol']} Max Pain Analysis",
            xaxis_title="Strike Price",
            yaxis_title="Total Pain ($)",
            height=500,
            hovermode='x'
        )
        
        return fig
    
    @staticmethod
    def plot_gamma_exposure(analysis_results: Dict) -> go.Figure:
        """Create gamma exposure visualization"""
        if 'gamma_exposure' not in analysis_results or analysis_results['gamma_exposure'].empty:
            return go.Figure()
        
        gamma_df = analysis_results['gamma_exposure']
        spot_price = analysis_results['spot_price']
        
        fig = go.Figure()
        
        # Plot net gamma exposure
        colors = ['red' if x < 0 else 'green' for x in gamma_df['net_gamma']]
        
        fig.add_trace(go.Bar(
            x=gamma_df['strike'],
            y=gamma_df['net_gamma'],
            name='Net Gamma Exposure',
            marker_color=colors
        ))
        
        # Add spot price line
        fig.add_vline(x=spot_price, line_dash="dash", line_color="blue",
                      annotation_text=f"Spot: {spot_price:.2f}")
        
        # Add zero line
        fig.add_hline(y=0, line_color="black", line_width=1)
        
        fig.update_layout(
            title=f"{analysis_results['symbol']} Gamma Exposure",
            xaxis_title="Strike Price",
            yaxis_title="Dollar Gamma Exposure",
            height=500,
            hovermode='x'
        )
        
        return fig
    
    @staticmethod
    def plot_volatility_surface_3d(surface_data: Dict, symbol: str) -> go.Figure:
        """
        Create 3D volatility surface plot

        Args:
            surface_data: Dictionary from VolatilitySurface.build_surface()
            symbol: Ticker symbol for title

        Returns:
            Plotly 3D surface figure
        """
        if 'error' in surface_data:
            return go.Figure()

        strikes = surface_data['grid_strikes']
        expiries = surface_data['grid_expiries']
        iv_grid = surface_data['grid_iv'] * 100  # Convert to percentage

        fig = go.Figure(data=[go.Surface(
            x=strikes,
            y=expiries,
            z=iv_grid,
            colorscale='Viridis',
            colorbar=dict(title="IV (%)"),
            hovertemplate='Strike: %{x:.2f}<br>DTE: %{y:.0f}<br>IV: %{z:.2f}%<extra></extra>'
        )])

        fig.update_layout(
            title=f'{symbol} Implied Volatility Surface',
            scene=dict(
                xaxis_title='Strike Price ($)',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2)
                )
            ),
            height=700,
            width=900
        )

        return fig

    @staticmethod
    def plot_volatility_skew(surface_data: Dict, symbol: str) -> go.Figure:
        """
        Plot volatility skew across expiries

        Args:
            surface_data: Dictionary from VolatilitySurface.build_surface()
            symbol: Ticker symbol

        Returns:
            Plotly figure with skew analysis
        """
        if 'skew' not in surface_data or surface_data['skew'].empty:
            return go.Figure()

        skew_df = surface_data['skew']

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Volatility Skew by Expiry', 'Put vs Call IV'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )

        # Skew plot
        fig.add_trace(
            go.Scatter(
                x=skew_df['days_to_expiry'],
                y=skew_df['skew'] * 100,
                mode='markers+lines',
                name='Skew (Put - Call)',
                marker=dict(size=10, color='purple'),
                line=dict(width=2)
            ),
            row=1, col=1
        )

        # Put vs Call IV
        fig.add_trace(
            go.Scatter(
                x=skew_df['days_to_expiry'],
                y=skew_df['put_iv'] * 100,
                mode='markers+lines',
                name='Put IV',
                marker=dict(size=8, color='red')
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=skew_df['days_to_expiry'],
                y=skew_df['call_iv'] * 100,
                mode='markers+lines',
                name='Call IV',
                marker=dict(size=8, color='green')
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Days to Expiry", row=2, col=1)
        fig.update_yaxes(title_text="Skew (%)", row=1, col=1)
        fig.update_yaxes(title_text="IV (%)", row=2, col=1)

        fig.update_layout(
            title=f'{symbol} Volatility Skew Analysis',
            height=700,
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def plot_term_structure(surface_data: Dict, symbol: str) -> go.Figure:
        """
        Plot volatility term structure (ATM IV across expiries)

        Args:
            surface_data: Dictionary from VolatilitySurface.build_surface()
            symbol: Ticker symbol

        Returns:
            Plotly figure with term structure
        """
        if 'term_structure' not in surface_data or surface_data['term_structure'].empty:
            return go.Figure()

        term_df = surface_data['term_structure']

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=term_df['days_to_expiry'],
            y=term_df['atm_iv'] * 100,
            mode='markers+lines',
            name='ATM IV',
            marker=dict(size=10, color='blue'),
            line=dict(width=3),
            fill='tozeroy',
            fillcolor='rgba(0,100,200,0.2)'
        ))

        fig.update_layout(
            title=f'{symbol} Volatility Term Structure',
            xaxis_title='Days to Expiry',
            yaxis_title='ATM Implied Volatility (%)',
            height=500,
            hovermode='x'
        )

        return fig

    @staticmethod
    def plot_vix_spy_correlation(vix_spy_data: Dict) -> go.Figure:
        """
        Plot VIX vs SPY correlation analysis

        Args:
            vix_spy_data: Dictionary from VIXSPYAnalyzer.fetch_historical_data()

        Returns:
            Plotly figure with correlation analysis
        """
        if not vix_spy_data or 'spy_returns' not in vix_spy_data:
            return go.Figure()

        spy_returns = vix_spy_data['spy_returns'].dropna() * 100
        vix_changes = vix_spy_data['vix_changes'].dropna() * 100

        # Align the series
        common_index = spy_returns.index.intersection(vix_changes.index)
        spy_returns = spy_returns.loc[common_index]
        vix_changes = vix_changes.loc[common_index]

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'SPY Returns vs VIX Changes (Correlation: {vix_spy_data["spy_vix_correlation"]:.2f})',
                'VIX vs Realized Volatility'
            ),
            vertical_spacing=0.15
        )

        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=spy_returns,
                y=vix_changes,
                mode='markers',
                name='Daily Data',
                marker=dict(size=6, color='blue', opacity=0.6)
            ),
            row=1, col=1
        )

        # Add trend line
        if len(spy_returns) > 1:
            z = np.polyfit(spy_returns, vix_changes, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(spy_returns.min(), spy_returns.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )

        # VIX vs Realized Vol bar chart
        fig.add_trace(
            go.Bar(
                x=['Current VIX', 'Realized Vol', 'Vol Risk Premium'],
                y=[
                    vix_spy_data['current_vix'],
                    vix_spy_data['realized_vol_spy'],
                    vix_spy_data['vix_spread']
                ],
                marker_color=['orange', 'blue', 'green' if vix_spy_data['vix_spread'] > 0 else 'red']
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="SPY Returns (%)", row=1, col=1)
        fig.update_yaxes(title_text="VIX Change (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        fig.update_layout(
            title='VIX and SPY Correlation Analysis',
            height=800,
            showlegend=True
        )

        return fig

    @staticmethod
    def create_dashboard(spy_results: Dict, vix_results: Dict = None) -> go.Figure:
        """Create a comprehensive dashboard with multiple metrics"""
        # This would create a combined dashboard view
        # Implementation depends on specific requirements
        pass


def main():
    """Main execution function with volatility surface and VIX/SPY analysis"""
    print("=" * 80)
    print("VOLATILITY SURFACE ANALYZER - SPY & VIX")
    print("=" * 80)

    # Initialize all components
    analyzer = OptionsAnalyzer()
    visualizer = OptionsVisualizer()
    vol_surface = VolatilitySurface()
    vix_spy_analyzer = VIXSPYAnalyzer()

    # === SPY Options Analysis ===
    print("\n[1/4] Fetching SPY options data...")
    spy_results = analyzer.analyze_symbol('SPY', num_expiries=5)

    # === Build Volatility Surface ===
    print("[2/4] Building volatility surface...")
    if 'options_data' in spy_results:
        surface_data = vol_surface.build_surface(
            spy_results['options_data'],
            spy_results['spot_price']
        )
        spy_results['vol_surface'] = surface_data
    else:
        surface_data = {}

    # === VIX/SPY Correlation Analysis ===
    print("[3/4] Analyzing VIX/SPY correlation...")
    vix_spy_data = vix_spy_analyzer.fetch_historical_data(days_back=30)
    vol_risk_premium = vix_spy_analyzer.calculate_volatility_risk_premium()

    # === Print Comprehensive Summary ===
    print("[4/4] Analysis complete!\n")
    print("=" * 80)
    print("SPY OPTIONS SUMMARY")
    print("=" * 80)

    if 'summary_stats' in spy_results:
        print(f"\n📊 Market Data:")
        print(f"   Spot Price: ${spy_results['spot_price']:.2f}")
        print(f"   Max Pain: ${spy_results['max_pain_strike']:.2f}")
        print(f"   Put/Call Volume Ratio: {spy_results['pc_volume_ratio']:.2f}")
        print(f"   Put/Call OI Ratio: {spy_results['pc_oi_ratio']:.2f}")

        print(f"\n📈 Key Levels:")
        print(f"   Support: {spy_results['key_levels']['support']}")
        print(f"   Resistance: {spy_results['key_levels']['resistance']}")

        stats = spy_results['summary_stats']
        print(f"\n📋 Trading Activity:")
        print(f"   Total Volume: {stats['total_volume']:,.0f}")
        print(f"   Total Open Interest: {stats['total_open_interest']:,.0f}")
        print(f"   Average IV: {stats['avg_implied_volatility']*100:.1f}%")
        print(f"   Number of Strikes: {stats['num_strikes']}")
        print(f"   Number of Expiries: {stats['num_expiries']}")

    # Volatility Surface Metrics
    if 'vol_surface' in spy_results and 'error' not in surface_data:
        print(f"\n🌊 Volatility Surface:")
        term_struct = surface_data.get('term_structure', pd.DataFrame())
        if not term_struct.empty:
            near_term_iv = term_struct.iloc[0]['atm_iv'] * 100
            far_term_iv = term_struct.iloc[-1]['atm_iv'] * 100
            print(f"   Near-term ATM IV: {near_term_iv:.2f}%")
            print(f"   Far-term ATM IV: {far_term_iv:.2f}%")
            print(f"   Term Structure: {'Contango' if far_term_iv > near_term_iv else 'Backwardation'}")

        skew = surface_data.get('skew', pd.DataFrame())
        if not skew.empty:
            avg_skew = skew['skew'].mean() * 100
            print(f"   Average Skew: {avg_skew:.2f}% (Put IV - Call IV)")

    # VIX/SPY Metrics
    if vix_spy_data:
        print(f"\n⚡ VIX/SPY Analysis:")
        print(f"   Current VIX: {vix_spy_data['current_vix']:.2f}")
        print(f"   SPY Realized Vol (30d): {vix_spy_data['realized_vol_spy']:.2f}%")
        print(f"   Volatility Risk Premium: {vol_risk_premium:.2f}%")
        print(f"   SPY-VIX Correlation: {vix_spy_data['spy_vix_correlation']:.3f}")

        if vol_risk_premium > 5:
            print(f"   💡 Signal: HIGH vol premium - consider selling volatility")
        elif vol_risk_premium < -5:
            print(f"   💡 Signal: NEGATIVE vol premium - consider buying volatility")
        else:
            print(f"   💡 Signal: Neutral vol premium")

    print("\n" + "=" * 80)

    # === Create All Visualizations ===
    print("\n📊 Generating visualizations...")

    # Original plots
    chain_fig = visualizer.plot_options_chain(spy_results)
    pain_fig = visualizer.plot_max_pain(spy_results)
    gamma_fig = visualizer.plot_gamma_exposure(spy_results)

    # New volatility surface plots
    if 'vol_surface' in spy_results and 'error' not in surface_data:
        surface_3d_fig = visualizer.plot_volatility_surface_3d(surface_data, 'SPY')
        skew_fig = visualizer.plot_volatility_skew(surface_data, 'SPY')
        term_structure_fig = visualizer.plot_term_structure(surface_data, 'SPY')
    else:
        surface_3d_fig = skew_fig = term_structure_fig = None

    # VIX/SPY correlation plot
    vix_spy_fig = visualizer.plot_vix_spy_correlation(vix_spy_data)

    # Show plots (uncomment to display in browser)
    print("\n💡 To display plots, uncomment the .show() lines in the code")
    # chain_fig.show()
    # pain_fig.show()
    # gamma_fig.show()
    # surface_3d_fig.show()
    # skew_fig.show()
    # term_structure_fig.show()
    # vix_spy_fig.show()

    # Save plots to HTML files
    print("\n💾 Saving plots to HTML files...")
    output_files = []

    chain_fig.write_html('spy_options_chain.html')
    output_files.append('spy_options_chain.html')

    pain_fig.write_html('spy_max_pain.html')
    output_files.append('spy_max_pain.html')

    if gamma_fig and hasattr(gamma_fig, 'data') and len(gamma_fig.data) > 0:
        gamma_fig.write_html('spy_gamma_exposure.html')
        output_files.append('spy_gamma_exposure.html')

    if surface_3d_fig:
        surface_3d_fig.write_html('spy_vol_surface_3d.html')
        output_files.append('spy_vol_surface_3d.html')

    if skew_fig:
        skew_fig.write_html('spy_vol_skew.html')
        output_files.append('spy_vol_skew.html')

    if term_structure_fig:
        term_structure_fig.write_html('spy_term_structure.html')
        output_files.append('spy_term_structure.html')

    vix_spy_fig.write_html('vix_spy_correlation.html')
    output_files.append('vix_spy_correlation.html')

    print(f"✅ Saved {len(output_files)} HTML visualization files:")
    for file in output_files:
        print(f"   - {file}")

    print("\n" + "=" * 80)
    print("✨ Analysis complete! Open the HTML files to view interactive plots.")
    print("=" * 80)

    return {
        'spy_results': spy_results,
        'surface_data': surface_data,
        'vix_spy_data': vix_spy_data,
        'vol_risk_premium': vol_risk_premium,
        'figures': {
            'chain': chain_fig,
            'pain': pain_fig,
            'gamma': gamma_fig,
            'surface_3d': surface_3d_fig,
            'skew': skew_fig,
            'term_structure': term_structure_fig,
            'vix_spy': vix_spy_fig
        }
    }


if __name__ == "__main__":
    results = main()

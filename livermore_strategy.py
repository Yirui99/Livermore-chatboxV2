"""
Livermore Trading Strategy Implementation

Backtest implementation based on Jesse Livermore trading strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

# Set of valid stock codes
# Only stock codes in this set are considered valid
VALID_STOCK_CODES = {
    "META",   # Meta Platforms Inc.
    "TSLA",   # Tesla Inc.
    "NVDA",   # NVIDIA Corporation
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "AMZN",   # Amazon.com Inc.
    "GOOGL",  # Alphabet Inc.
    # Add more valid stock codes here as needed
}


def is_valid_stock_code(stock_symbol: str) -> tuple[bool, str]:
    """
    Check if stock code is in the valid stock codes set
    
    Args:
        stock_symbol: Stock ticker symbol
    
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    # Normalize stock symbol (uppercase, strip whitespace)
    normalized_symbol = stock_symbol.strip().upper()
    
    # Check if stock code is in the valid set
    if normalized_symbol not in VALID_STOCK_CODES:
        return False, f"❌ Stock code '{normalized_symbol}' is not in the valid stock codes list. Please use a valid stock code."
    
    return True, None


def download_stock_data(stock_symbol: str, start_date: datetime, end_date: datetime):
    """
    Download stock data
    
    Args:
        stock_symbol: Stock ticker symbol
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame: Stock data, None if failed
        str: Error message, None if successful
    """
    try:
        # Normalize stock symbol
        stock_symbol = stock_symbol.strip().upper()
        
        # Validate stock code against valid set
        is_valid, error_msg = is_valid_stock_code(stock_symbol)
        if not is_valid:
            return None, error_msg
        
        # Pre-validate stock code format (single letters are usually not valid)
        if len(stock_symbol) == 1:
            return None, f"❌ Stock code '{stock_symbol}' format is invalid. Stock codes typically require at least 2 characters. Please check if the stock code is correct."
        
        # Try to download data
        stock = yf.download(stock_symbol, start_date, end_date, progress=False)
        
        # Check if empty or no data
        if stock.empty:
            return None, f"❌ Stock code '{stock_symbol}' does not exist or data cannot be retrieved. Please check if the stock code is correct, or try other stock codes (e.g., AAPL, TSLA, MSFT, etc.)."
        
        # Handle multi-level column index
        if stock.columns.nlevels > 1:
            # Check if stock code is in columns
            available_tickers = stock.columns.get_level_values('Ticker').unique().tolist()
            if stock_symbol not in available_tickers:
                return None, f"❌ Stock code '{stock_symbol}' does not exist. Available stock codes: {', '.join(available_tickers)}"
            df = stock.xs(stock_symbol, axis=1, level='Ticker')
        else:
            df = stock.copy()
        
        # Ensure necessary columns exist
        if 'Close' not in df.columns:
            return None, f"❌ Stock code '{stock_symbol}' data format error: missing closing price data"
        
        # Check if data is empty
        if df.empty or len(df) == 0:
            return None, f"❌ Stock code '{stock_symbol}' has no data in the specified time range. Please try adjusting the time range."
        
        # Check if there is valid price data (not all NaN)
        if df['Close'].isna().all():
            return None, f"❌ Stock code '{stock_symbol}' does not exist or cannot retrieve valid data. Please check if the stock code is correct."
        
        # Check valid data point count (need at least some valid data)
        valid_data = df['Close'].dropna()
        valid_data_count = len(valid_data)
        
        if valid_data_count == 0:
            return None, f"❌ Stock code '{stock_symbol}' does not exist or cannot retrieve valid data. Please check if the stock code is correct."
        
        # Check if price data is reasonable (prices should be greater than 0)
        if (valid_data <= 0).any():
            return None, f"❌ Stock code '{stock_symbol}' data is abnormal (contains invalid prices). Stock code may not exist."
        
        # Check if data is too little (less than 10 trading days may be abnormal unless time range is very short)
        # But if time range is long (over 1 year) and data is little, it's definitely abnormal
        date_range_days = (end_date - start_date).days
        if date_range_days > 365 and valid_data_count < 50:
            return None, f"❌ Stock code '{stock_symbol}' does not exist or data is abnormal. Only found {valid_data_count} valid data points in a {date_range_days}-day time range, which usually indicates an invalid stock code."
        elif valid_data_count < 10:
            return None, f"❌ Stock code '{stock_symbol}' has insufficient data (only {valid_data_count} valid data points). Stock code may not exist or data is abnormal."
        
        return df, None
        
    except Exception as e:
        error_msg = str(e).lower()
        if "no data" in error_msg or "invalid" in error_msg or "not found" in error_msg:
            return None, f"❌ Stock code '{stock_symbol}' does not exist or is invalid. Please check if the stock code is correct."
        return None, f"❌ Error downloading data: {str(e)}"


def calculate_indicators(df: pd.DataFrame, sma_short: int = 50, sma_long: int = 200, breakout_window: int = 20):
    """
    Calculate technical indicators
    
    Args:
        df: Stock data DataFrame
        sma_short: Short-term moving average window (default 50)
        sma_long: Long-term moving average window (default 200)
        breakout_window: Breakout window (default 20)
    
    Returns:
        DataFrame: DataFrame with technical indicators added
    """
    # Calculate moving averages
    df[f'{sma_short}MA'] = df['Close'].rolling(window=sma_short).mean()
    df[f'{sma_long}MA'] = df['Close'].rolling(window=sma_long).mean()
    
    # Define breakout levels (highs and lows of past N days)
    df[f'{breakout_window}High'] = df['Close'].rolling(window=breakout_window).max()
    df[f'{breakout_window}Low'] = df['Close'].rolling(window=breakout_window).min()
    
    return df


def generate_signals(df: pd.DataFrame, breakout_window: int = 20):
    """
    Generate trading signals
    
    Args:
        df: DataFrame with technical indicators
        breakout_window: Breakout window (default 20)
    
    Returns:
        DataFrame: DataFrame with trading signals added
    """
    # Initialize signals and positions
    df['Signal'] = 0
    df['Position'] = np.nan
    
    # Buy when:
    # - Price breaks above past N-day high
    # - Price is above both 50MA and 200MA
    df.loc[
        (df['Close'] > df[f'{breakout_window}High'].shift(1)) &
        (df['Close'] > df['50MA']) &
        (df['Close'] > df['200MA']),
        'Signal'
    ] = 1
    
    # Sell when:
    # - Price drops below past N-day low
    sell_signal = df['Close'] < df[f'{breakout_window}Low'].shift(1)
    df.loc[sell_signal, 'Signal'] = -1
    
    # Generate positions based on trading signals (long-only)
    df.loc[df['Signal'] == 1, 'Position'] = 1   # Buy signal → take a long position
    df.loc[df['Signal'] == -1, 'Position'] = 0  # Sell signal → close the position
    
    # Forward fill positions to maintain the same position until the next signal
    df['Position'] = df['Position'].ffill().fillna(0)
    
    return df


def calculate_returns(df: pd.DataFrame):
    """
    Calculate returns
    
    Args:
        df: DataFrame with position signals
    
    Returns:
        DataFrame: DataFrame with return calculations added
    """
    # Calculate returns
    df['Buy-and-Hold Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Buy-and-Hold Return'] * df['Position']
    
    # Calculate cumulative returns (using cumsum to match notebook implementation)
    df['Buy-and-Hold Cumulative'] = df['Buy-and-Hold Return'].cumsum()
    df['Strategy Cumulative'] = df['Strategy Return'].cumsum()
    
    return df


def run_livermore_strategy(stock_symbol: str, start_date: datetime, end_date: datetime, 
                          sma_short: int = 50, sma_long: int = 200, breakout_window: int = 20):
    """
    Run Livermore strategy backtest
    
    Args:
        stock_symbol: Stock ticker symbol
        start_date: Start date
        end_date: End date
        sma_short: Short-term moving average window (default 50)
        sma_long: Long-term moving average window (default 200)
        breakout_window: Breakout window (default 20)
    
    Returns:
        DataFrame: DataFrame with strategy results, None if failed
        str: Error message, None if successful
    """
    try:
        # Normalize stock symbol
        stock_symbol = stock_symbol.strip().upper()
        
        # Validate stock code against valid set first
        is_valid, error_msg = is_valid_stock_code(stock_symbol)
        if not is_valid:
            return None, error_msg
        
        # Download stock data
        df, error = download_stock_data(stock_symbol, start_date, end_date)
        if error:
            return None, error
        
        # Calculate technical indicators
        df = calculate_indicators(df, sma_short, sma_long, breakout_window)
        
        # Generate trading signals
        df = generate_signals(df, breakout_window)
        
        # Calculate returns
        df = calculate_returns(df)
        
        return df, None
        
    except Exception as e:
        return None, f"Strategy execution error: {str(e)}"


def get_strategy_summary(df: pd.DataFrame):
    """
    Get strategy backtest summary
    
    Args:
        df: DataFrame with strategy results
    
    Returns:
        dict: Dictionary containing various statistical metrics
    """
    if df is None or df.empty:
        return None
    
    strategy_return = df['Strategy Cumulative'].iloc[-1] if not df['Strategy Cumulative'].empty else 0
    buyhold_return = df['Buy-and-Hold Cumulative'].iloc[-1] if not df['Buy-and-Hold Cumulative'].empty else 0
    
    improvement = ((strategy_return - buyhold_return) / abs(buyhold_return) * 100) if buyhold_return != 0 else 0
    
    # Get trading signals
    signals = df[df['Position'] != df['Position'].shift(1)].copy()
    signals = signals[signals['Position'] != 0]
    
    return {
        'strategy_return': strategy_return,
        'buyhold_return': buyhold_return,
        'improvement': improvement,
        'signals': signals
    }


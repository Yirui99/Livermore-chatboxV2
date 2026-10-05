"""
Enhanced Livermore Trading Strategy with Additional Technical Indicators

Adds: RSI, MACD, Bollinger Bands, ATR (Average True Range), Volume MA
Provides both original and enhanced strategy for comparison.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from livermore_strategy import download_stock_data, VALID_STOCK_CODES


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands"""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — measures volatility"""
    high = df['High'] if 'High' in df.columns else df['Close']
    low = df['Low'] if 'Low' in df.columns else df['Close']
    close_prev = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def calculate_volume_ma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume Moving Average"""
    if 'Volume' in df.columns:
        return df['Volume'].rolling(window=window).mean()
    return pd.Series(np.nan, index=df.index)


def run_enhanced_strategy(stock_symbol: str, start_date: datetime, end_date: datetime,
                          sma_short: int = 50, sma_long: int = 200,
                          breakout_window: int = 20, rsi_window: int = 14):
    """
    Enhanced Livermore strategy with multi-indicator confirmation.

    Buy when ALL of:
      1. Price breaks above N-day high (original)
      2. Price > 50MA AND > 200MA (original)
      3. RSI > 50 (momentum confirmation — not overbought yet)
      4. MACD histogram > 0 (bullish momentum)
      5. Volume > Volume MA (volume confirmation)

    Sell when ANY of:
      1. Price drops below N-day low (original)
      2. RSI > 70 AND MACD histogram turns negative (overbought reversal)
      3. Price drops below lower Bollinger Band (extreme weakness)

    ATR-based trailing stop:
      - Stop loss = Close - 2 * ATR (adaptive to volatility)
    """
    stock_symbol = stock_symbol.strip().upper()

    df, error = download_stock_data(stock_symbol, start_date, end_date)
    if error:
        return None, None, error

    # ---- Original indicators ----
    df[f'{sma_short}MA'] = df['Close'].rolling(window=sma_short).mean()
    df[f'{sma_long}MA'] = df['Close'].rolling(window=sma_long).mean()
    df[f'{breakout_window}High'] = df['Close'].rolling(window=breakout_window).max()
    df[f'{breakout_window}Low'] = df['Close'].rolling(window=breakout_window).min()

    # ---- New indicators ----
    df['RSI'] = calculate_rsi(df['Close'], rsi_window)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df)
    df['Volume_MA'] = calculate_volume_ma(df)

    # ========== ORIGINAL STRATEGY ==========
    df['Signal_Orig'] = 0
    df['Position_Orig'] = np.nan

    df.loc[
        (df['Close'] > df[f'{breakout_window}High'].shift(1)) &
        (df['Close'] > df[f'{sma_short}MA']) &
        (df['Close'] > df[f'{sma_long}MA']),
        'Signal_Orig'
    ] = 1

    df.loc[
        df['Close'] < df[f'{breakout_window}Low'].shift(1),
        'Signal_Orig'
    ] = -1

    df.loc[df['Signal_Orig'] == 1, 'Position_Orig'] = 1
    df.loc[df['Signal_Orig'] == -1, 'Position_Orig'] = 0
    df['Position_Orig'] = df['Position_Orig'].ffill().fillna(0)

    # ========== ENHANCED STRATEGY ==========
    df['Signal_Enhanced'] = 0
    df['Position_Enhanced'] = np.nan

    # Enhanced buy: original + RSI + MACD + Volume confirmation
    has_volume = 'Volume' in df.columns and not df['Volume'].isna().all()

    buy_condition = (
        (df['Close'] > df[f'{breakout_window}High'].shift(1)) &
        (df['Close'] > df[f'{sma_short}MA']) &
        (df['Close'] > df[f'{sma_long}MA']) &
        (df['RSI'] > 50) & (df['RSI'] < 75) &  # Momentum but not overbought
        (df['MACD_Hist'] > 0)                    # Bullish MACD
    )
    if has_volume:
        buy_condition = buy_condition & (df['Volume'] > df['Volume_MA'])

    df.loc[buy_condition, 'Signal_Enhanced'] = 1

    # Enhanced sell: original OR overbought reversal OR Bollinger breakdown
    sell_condition = (
        (df['Close'] < df[f'{breakout_window}Low'].shift(1)) |  # Original
        ((df['RSI'] > 70) & (df['MACD_Hist'] < 0)) |             # Overbought reversal
        (df['Close'] < df['BB_Lower'])                            # Extreme weakness
    )
    df.loc[sell_condition, 'Signal_Enhanced'] = -1

    df.loc[df['Signal_Enhanced'] == 1, 'Position_Enhanced'] = 1
    df.loc[df['Signal_Enhanced'] == -1, 'Position_Enhanced'] = 0
    df['Position_Enhanced'] = df['Position_Enhanced'].ffill().fillna(0)

    # ---- ATR trailing stop ----
    df['ATR_Stop'] = df['Close'] - 2 * df['ATR']

    # Apply ATR stop: if in position and close drops below ATR stop, exit
    for i in range(1, len(df)):
        if df['Position_Enhanced'].iloc[i] == 1:
            if df['Close'].iloc[i] < df['ATR_Stop'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('Position_Enhanced')] = 0

    # ========== RETURNS ==========
    df['Daily_Return'] = df['Close'].pct_change()

    df['Return_BuyHold'] = df['Daily_Return'].cumsum()
    df['Return_Original'] = (df['Daily_Return'] * df['Position_Orig']).cumsum()
    df['Return_Enhanced'] = (df['Daily_Return'] * df['Position_Enhanced']).cumsum()

    return df, stock_symbol, None


def get_comparison_summary(df: pd.DataFrame, symbol: str):
    """Generate comparison summary between original and enhanced strategies."""
    if df is None or df.empty:
        return None

    buyhold = df['Return_BuyHold'].iloc[-1]
    orig = df['Return_Original'].iloc[-1]
    enhanced = df['Return_Enhanced'].iloc[-1]

    orig_signals = len(df[df['Signal_Orig'] != 0])
    enhanced_signals = len(df[df['Signal_Enhanced'] != 0])

    # Max drawdown calculation
    def max_drawdown(cumret):
        peak = cumret.cummax()
        dd = cumret - peak
        return dd.min()

    mdd_buyhold = max_drawdown(df['Return_BuyHold'])
    mdd_orig = max_drawdown(df['Return_Original'])
    mdd_enhanced = max_drawdown(df['Return_Enhanced'])

    # Sharpe ratio (annualized, assuming 252 trading days)
    def sharpe(daily_returns):
        if daily_returns.std() == 0:
            return 0
        return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    sharpe_buyhold = sharpe(df['Daily_Return'].dropna())
    sharpe_orig = sharpe((df['Daily_Return'] * df['Position_Orig']).dropna())
    sharpe_enhanced = sharpe((df['Daily_Return'] * df['Position_Enhanced']).dropna())

    return {
        'symbol': symbol,
        'buyhold_return': buyhold,
        'original_return': orig,
        'enhanced_return': enhanced,
        'orig_signals': orig_signals,
        'enhanced_signals': enhanced_signals,
        'mdd_buyhold': mdd_buyhold,
        'mdd_original': mdd_orig,
        'mdd_enhanced': mdd_enhanced,
        'sharpe_buyhold': sharpe_buyhold,
        'sharpe_original': sharpe_orig,
        'sharpe_enhanced': sharpe_enhanced,
    }


def main():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)

    stocks = ["AAPL", "TSLA", "NVDA", "MSFT", "META", "AMZN", "GOOGL"]

    print("=" * 120)
    print("ENHANCED LIVERMORE STRATEGY — MULTI-INDICATOR BACKTEST")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Added indicators: RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14), Volume MA(20)")
    print("=" * 120)

    results = []
    for sym in stocks:
        df, symbol, error = run_enhanced_strategy(sym, start_date, end_date)
        if error:
            print(f"  {sym}: {error}")
            continue
        summary = get_comparison_summary(df, symbol)
        if summary:
            results.append(summary)

    # Print comparison table
    print(f"\n{'Symbol':<8} | {'Buy&Hold':>10} | {'Original':>10} | {'Enhanced':>10} | "
          f"{'MDD B&H':>8} | {'MDD Orig':>8} | {'MDD Enh':>8} | "
          f"{'Sharpe BH':>9} | {'Sharpe Orig':>11} | {'Sharpe Enh':>10} | "
          f"{'Sig Orig':>8} | {'Sig Enh':>7}")
    print("-" * 140)

    for r in results:
        print(f"{r['symbol']:<8} | "
              f"{r['buyhold_return']:>9.2%} | "
              f"{r['original_return']:>9.2%} | "
              f"{r['enhanced_return']:>9.2%} | "
              f"{r['mdd_buyhold']:>7.2%} | "
              f"{r['mdd_original']:>7.2%} | "
              f"{r['mdd_enhanced']:>7.2%} | "
              f"{r['sharpe_buyhold']:>9.3f} | "
              f"{r['sharpe_original']:>11.3f} | "
              f"{r['sharpe_enhanced']:>10.3f} | "
              f"{r['orig_signals']:>8} | "
              f"{r['enhanced_signals']:>7}")

    # Summary
    if results:
        print("\n" + "=" * 120)
        print("KEY FINDINGS:")

        avg_mdd_orig = np.mean([r['mdd_original'] for r in results])
        avg_mdd_enh = np.mean([r['mdd_enhanced'] for r in results])
        avg_sharpe_orig = np.mean([r['sharpe_original'] for r in results])
        avg_sharpe_enh = np.mean([r['sharpe_enhanced'] for r in results])
        avg_signals_orig = np.mean([r['orig_signals'] for r in results])
        avg_signals_enh = np.mean([r['enhanced_signals'] for r in results])

        wins = sum(1 for r in results if r['enhanced_return'] > r['original_return'])
        mdd_wins = sum(1 for r in results if r['mdd_enhanced'] > r['mdd_original'])

        print(f"  Enhanced strategy beats original:    {wins}/{len(results)} stocks")
        print(f"  Enhanced has smaller max drawdown:   {mdd_wins}/{len(results)} stocks")
        print(f"  Avg Sharpe (original):   {avg_sharpe_orig:.3f}")
        print(f"  Avg Sharpe (enhanced):   {avg_sharpe_enh:.3f}")
        print(f"  Avg Max Drawdown (orig): {avg_mdd_orig:.2%}")
        print(f"  Avg Max Drawdown (enh):  {avg_mdd_enh:.2%}")
        print(f"  Avg signals (orig):      {avg_signals_orig:.0f}")
        print(f"  Avg signals (enhanced):  {avg_signals_enh:.0f}")
        print("=" * 120)


if __name__ == "__main__":
    main()

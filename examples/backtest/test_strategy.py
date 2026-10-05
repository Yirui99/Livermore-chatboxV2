import sys
import os
from datetime import datetime

# Add the directory to sys.path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from livermore_strategy import run_livermore_strategy, get_strategy_summary

def test_stock(symbol):
    print(f"--- Testing {symbol} ---")
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    df, error = run_livermore_strategy(symbol, start_date, end_date)
    if error:
        print(f"Error: {error}")
        return
        
    summary = get_strategy_summary(df)
    
    print(f"Strategy Return: {summary['strategy_return']:.2%}")
    print(f"Buy & Hold Return: {summary['buyhold_return']:.2%}")
    print(f"Improvement: {summary['improvement']:.2f}%")
    print(f"Number of signals/trades: {len(summary['signals'])}")
    print()

if __name__ == "__main__":
    test_stock("AAPL")
    test_stock("TSLA")
    test_stock("NVDA")
    test_stock("MSFT")

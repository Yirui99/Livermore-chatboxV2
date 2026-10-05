# pages/stock_selection_page.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from livermore_strategy import run_livermore_strategy, get_strategy_summary


def show_stock_selection_page():
    """Display stock selection page"""

    # Initialize color scheme setting
    if "color_scheme" not in st.session_state:
        st.session_state.color_scheme = "green_up"  # Default: green for positive/up

    # Add Navy Blue background styling and Gold text
    # Also add color scheme CSS
    color_scheme_css = ""
    if st.session_state.color_scheme == "red_up":
        color_scheme_css = """
            /* Invert metric colors: Red for positive, Green for negative */
            [data-testid="stMetric"] [data-testid="stMetricDelta"] {
                color: inherit !important;
            }
            /* Red for positive values (up arrow or positive text) */
            [data-testid="stMetric"] [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]),
            [data-testid="stMetric"] [data-testid="stMetricDelta"]:not(:has(svg[data-testid="stMetricDeltaIcon-Down"])):not(:empty) {
                color: #ff4b4b !important;
            }
            [data-testid="stMetric"] [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"] {
                color: #ff4b4b !important;
                fill: #ff4b4b !important;
            }
            /* Green for negative values (down arrow or negative text) */
            [data-testid="stMetric"] [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {
                color: #00cc88 !important;
            }
            [data-testid="stMetric"] [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"] {
                color: #00cc88 !important;
                fill: #00cc88 !important;
            }
        """

    st.markdown("""
        <style>
        /* Hide sidebar on stock selection page */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Remove left padding/margin when sidebar is hidden */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        div[data-baseweb="popover"] div, 
        div[data-baseweb="popover"] span,
        div[data-baseweb="popover"] button {
            color: #000000 !important;
        }
        div[data-baseweb="calendar"] button[data-baseweb="calendar-day"] {
            color: #000000 !important;
            font-weight: bold !important;
        }
        div[role="columnheader"] {
            color: #000000 !important;
            font-weight: 900 !important;
        }
        div[data-baseweb="calendar"] button svg {
            fill: #000000 !important; /* 箭头颜色 */
        }
        div[data-baseweb="calendar"] button[aria-selected="true"],
        div[data-baseweb="calendar"] button[aria-selected="true"]:hover {
            color: #ffffff !important;
            background-color: #ff4b4b !important; /* 保持红色背景 */
        }
        [data-testid="stDateInput"] div[data-baseweb="popover"] p,
        [data-testid="stDateInput"] div[data-baseweb="popover"] div {
            color: black !important;
        }
        
        /* All titles and headers - Gold color, clear and visible */
        h1, h2, h3, h4, h5, h6 {
            color: #d4af37 !important;
            font-weight: bold !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Ensure all title elements are gold */
        .main h1,
        .main h2,
        .main h3,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        div[data-testid="stAppViewContainer"] h1,
        div[data-testid="stAppViewContainer"] h2,
        div[data-testid="stAppViewContainer"] h3 {
            color: #d4af37 !important;
            font-weight: bold !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Info box text - Gold color */
        [data-testid="stInfo"],
        [data-testid="stInfo"] *,
        [data-testid="stInfo"] p,
        [data-testid="stInfo"] div,
        [data-testid="stInfo"] span {
            color: #d4af37 !important;
            font-weight: bold !important;
        }
        
        /* Input field labels - Gold color */
        .stTextInput label,
        .stTextInput > label,
        .stTextInput label p,
        .stTextInput label span,
        .stTextInput label div,
        .stSelectbox label,
        .stSelectbox > label,
        .stSelectbox label p,
        .stSelectbox label span,
        .stSelectbox label div,
        .stDateInput label,
        .stDateInput > label,
        .stDateInput label p,
        .stDateInput label span,
        .stDateInput label div,
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stDateInput"] label {
            color: #d4af37 !important;
            font-weight: bold !important;
        }
        
        /* Radio button label and options - White and bold */
        .stRadio label,
        .stRadio > label,
        .stRadio label p,
        .stRadio label span,
        .stRadio label div,
        .stRadio [data-testid="stMarkdownContainer"] p,
        .stRadio [data-testid="stMarkdownContainer"] span,
        .stRadio [data-testid="stMarkdownContainer"] div,
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] > label,
        [data-testid="stRadio"] label * {
            color: #ffffff !important;
            font-weight: bold !important;
        }
        
        /* Markdown text in main content - Gold color */
        .main .stMarkdown,
        .main .stMarkdown p,
        .main .stMarkdown p *,
        .main .stMarkdown ul,
        .main .stMarkdown ul li,
        .main .stMarkdown ul li *,
        .main .stMarkdown strong,
        .main .stMarkdown strong *,
        [data-testid="stAppViewContainer"] .stMarkdown,
        [data-testid="stAppViewContainer"] .stMarkdown p,
        [data-testid="stAppViewContainer"] .stMarkdown p *,
        [data-testid="stAppViewContainer"] .stMarkdown ul,
        [data-testid="stAppViewContainer"] .stMarkdown ul li,
        [data-testid="stAppViewContainer"] .stMarkdown ul li *,
        [data-testid="stAppViewContainer"] .stMarkdown strong,
        [data-testid="stAppViewContainer"] .stMarkdown strong * {
            color: #d4af37 !important;
        }
        </style>
        """, unsafe_allow_html=True)


    st.title("Stock Selection & Analysis")

    # Back to main page button
    if st.button("← Back to Main"):
        st.session_state.page = "main"
        st.rerun()

    # Settings expander for color scheme
    with st.expander("⚙️ Settings", expanded=False):
        color_scheme = st.radio(
            "Color Scheme",
            options=["green_up", "red_up"],
            index=0 if st.session_state.color_scheme == "green_up" else 1,
            format_func=lambda
                x: "Green for Up/Profit (Red for Down/Loss)" if x == "green_up" else "Red for Up/Profit (Green for Down/Loss)",
            key="color_scheme_radio"
        )
        # Update and rerun if color scheme changed
        if color_scheme != st.session_state.color_scheme:
            st.session_state.color_scheme = color_scheme
            st.rerun()
        else:
            st.session_state.color_scheme = color_scheme

    st.markdown("---")

    # Stock selection area
    st.subheader("Select Stock")

    # Popular stocks list (Magnificent 7)
    popular_stocks = {
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "GOOGL": "Alphabet Inc."
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_stock = st.selectbox(
            "Select Stock Code",
            options=list(popular_stocks.keys()),
            format_func=lambda x: f"{x} - {popular_stocks[x]}"
        )

    with col2:
        custom_stock = st.text_input("Or enter custom stock code", placeholder="e.g., AAPL", key="custom_stock_input")

    # Validate custom stock input
    stock_error = None
    use_custom_stock = False

    if custom_stock:
        custom_stock_cleaned = custom_stock.strip().upper()
        if not custom_stock_cleaned:
            stock_error = "Stock code cannot be empty or contain only spaces"
        elif len(custom_stock_cleaned) == 1:
            stock_error = "Stock code must be at least 2 characters. Single letters (e.g., 'A') are not valid stock codes."
        elif len(custom_stock_cleaned) > 10:
            stock_error = "Stock code length cannot exceed 10 characters"
        elif not custom_stock_cleaned.replace('.', '').replace('-', '').isalnum():
            stock_error = "Stock code can only contain letters, numbers, dots, and hyphens"
        else:
            selected_stock = custom_stock_cleaned
            use_custom_stock = True

    # Display error message
    if stock_error:
        st.error(stock_error)
        # If there's an error, use the dropdown selection
        use_custom_stock = False

    # Time range selection
    st.markdown("---")
    st.subheader("Select Time Range")

    # Calendar JavaScript - Simple and direct approach
    st.markdown("""
    <script>
        (function() {
            function fixCalendar() {
                // Find all calendar buttons
                const buttons = document.querySelectorAll('[data-baseweb="calendar"] button, [data-baseweb="popover"] button');
                buttons.forEach(btn => {
                    const text = btn.textContent.trim();
                    // If it's a date number (1-31)
                    if (text && !isNaN(text) && parseInt(text) >= 1 && parseInt(text) <= 31) {
                        const parent = btn.closest('[role="gridcell"]');
                        const isSelected = parent && parent.getAttribute('aria-selected') === 'true';

                        if (!isSelected) {
                            // Force SOLID black, extra bold, larger, no shadow, full opacity
                            btn.style.setProperty('color', '#000000', 'important');
                            btn.style.setProperty('font-weight', '900', 'important');
                            btn.style.setProperty('font-size', '20px', 'important');
                            btn.style.setProperty('text-shadow', 'none', 'important');
                            btn.style.setProperty('opacity', '1', 'important');
                            btn.style.setProperty('-webkit-font-smoothing', 'antialiased', 'important');
                            btn.style.setProperty('-moz-osx-font-smoothing', 'grayscale', 'important');

                            // Also fix any spans inside
                            btn.querySelectorAll('span').forEach(span => {
                                span.style.setProperty('color', '#000000', 'important');
                                span.style.setProperty('font-weight', '900', 'important');
                                span.style.setProperty('opacity', '1', 'important');
                            });
                        }
                    }
                });

                // Fix weekday headers - SOLID black
                document.querySelectorAll('[role="columnheader"]').forEach(header => {
                    header.style.setProperty('color', '#000000', 'important');
                    header.style.setProperty('font-weight', 'bold', 'important');
                    header.style.setProperty('opacity', '1', 'important');
                    header.querySelectorAll('*').forEach(child => {
                        child.style.setProperty('color', '#000000', 'important');
                        child.style.setProperty('opacity', '1', 'important');
                    });
                });

                // Fix month/year headers
                document.querySelectorAll('[data-baseweb="header"] button, [data-baseweb="header"] span').forEach(el => {
                    el.style.setProperty('color', '#000000', 'important');
                    el.style.setProperty('font-weight', 'bold', 'important');
                });
            }

            // Run immediately
            fixCalendar();

            // Run on intervals
            setInterval(fixCalendar, 100);

            // Watch for DOM changes
            const observer = new MutationObserver(fixCalendar);
            observer.observe(document.body, { childList: true, subtree: true, attributes: true });

            // Run when clicking on date inputs
            document.addEventListener('click', function(e) {
                if (e.target.closest('.stDateInput') || e.target.closest('[data-baseweb="popover"]')) {
                    setTimeout(fixCalendar, 50);
                    setTimeout(fixCalendar, 150);
                    setTimeout(fixCalendar, 300);
                }
            }, true);
        })();
    </script>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2019, 1, 1).date(),
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime.today().date()
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.today().date(),
            min_value=start_date,
            max_value=datetime.today().date()
        )

    # Validate stock code
    if selected_stock and not stock_error:
        # Display current stock and time range
        stock_source = "Custom Input" if use_custom_stock else "Preset List"
        if use_custom_stock:
            # Custom input: Show input info, clearly state need for verification
            st.info(f"Entered Stock Code: **{selected_stock}** | Time Range: {start_date} to {end_date}")
            st.warning(
                "**Important**: Stock code format validation passed, but **stock existence has not been verified**. Please click the button below to run backtest for verification. If the stock code does not exist, an error will be displayed.")
        else:
            # Preset list: Show confirmation (preset stocks are usually valid)
            st.info(f"Selected Stock: **{selected_stock}** ({stock_source}) | Time Range: {start_date} to {end_date}")

        # Run backtest button
        if st.button("Run Backtest", type="primary", use_container_width=True):
            # Validate stock code format again
            final_stock = selected_stock.strip().upper() if selected_stock else None
            if not final_stock:
                st.error("Please select or enter a valid stock code")
            elif len(final_stock) == 1:
                st.error("Stock code must be at least 2 characters. Single letters are not valid stock codes.")
            elif len(final_stock) > 10:
                st.error("Stock code length cannot exceed 10 characters")
            elif not final_stock.replace('.', '').replace('-', '').isalnum():
                st.error("Stock code can only contain letters, numbers, dots, and hyphens")
            else:
                with st.spinner(f"Validating stock code '{final_stock}' and downloading data..."):
                    df, error = run_livermore_strategy(final_stock, start_date, end_date)

                if error:
                    # Display detailed error message
                    st.error(error)
                    # If stock code doesn't exist, provide additional help
                    if "does not exist" in error.lower() or "cannot" in error.lower() or "invalid" in error.lower() or "not found" in error.lower():
                        with st.expander("💡 How to Enter a Correct Stock Code?", expanded=False):
                            st.markdown("""
                            **Common Stock Code Examples:**
                            - **AAPL** - Apple Inc.
                            - **TSLA** - Tesla Inc.
                            - **MSFT** - Microsoft Corporation
                            - **GOOGL** - Alphabet Inc.
                            - **AMZN** - Amazon.com Inc.
                            - **META** - Meta Platforms Inc.
                            - **NVDA** - NVIDIA Corporation

                            **Tips:**
                            - Stock codes are usually 1-5 uppercase letters
                            - Make sure you enter a valid stock code, not numbers or other characters
                            - If the stock code doesn't exist, yfinance will not be able to retrieve data
                            - Pay attention to spelling: e.g., "AAPL" is correct, while "APPL" is wrong
                            """)
                elif df is not None:
                    # Only show confirmation after successfully getting data
                    st.success(f"Stock code '{final_stock}' validated successfully! Data loaded.")
                    st.markdown("---")
                    st.subheader("Backtest Results")

                    # Get strategy summary
                    summary = get_strategy_summary(df)

                    if summary:
                        # Helper function to get color based on value and color scheme
                        def get_color_class(value, is_improvement=False):
                            """Get color class based on value and color scheme"""
                            if value is None or (isinstance(value, (int, float)) and value == 0):
                                return ""

                            is_positive = value > 0

                            if st.session_state.color_scheme == "red_up":
                                # Red for positive, green for negative
                                return "metric-positive-red" if is_positive else "metric-negative-green"
                            else:
                                # Green for positive, red for negative (default)
                                return "metric-positive-green" if is_positive else "metric-negative-red"

                        # Helper function to get arrow symbol
                        def get_arrow(value):
                            """Get arrow symbol based on value"""
                            if value is None or (isinstance(value, (int, float)) and value == 0):
                                return ""
                            return "↑" if value > 0 else "↓"

                        # Display statistics using custom HTML
                        col1, col2, col3 = st.columns(3)

                        # Buy-and-Hold Return
                        buyhold_return = summary['buyhold_return']
                        buyhold_color = get_color_class(buyhold_return)
                        buyhold_arrow = get_arrow(buyhold_return)

                        with col1:
                            st.markdown(f"""
                            <div style="padding: 1rem; background-color: rgba(26, 35, 50, 0.5); border-radius: 0.5rem;">
                                <div style="font-size: 0.875rem; color: #f5f5f5; margin-bottom: 0.5rem;">Buy-and-Hold Return</div>
                                <div style="font-size: 2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.25rem;">{buyhold_return:.2%}</div>
                                <div class="{buyhold_color}" style="font-size: 0.875rem; font-weight: bold;">
                                    {buyhold_arrow} {buyhold_return:.2%}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Strategy Return
                        strategy_return = summary['strategy_return']
                        strategy_delta = strategy_return - summary['buyhold_return']
                        strategy_color = get_color_class(strategy_delta)
                        strategy_arrow = get_arrow(strategy_delta)

                        with col2:
                            st.markdown(f"""
                            <div style="padding: 1rem; background-color: rgba(26, 35, 50, 0.5); border-radius: 0.5rem;">
                                <div style="font-size: 0.875rem; color: #f5f5f5; margin-bottom: 0.5rem;">Livermore Strategy Return</div>
                                <div style="font-size: 2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.25rem;">{strategy_return:.2%}</div>
                                <div class="{strategy_color}" style="font-size: 0.875rem; font-weight: bold;">
                                    {strategy_arrow} {strategy_delta:.2%}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Strategy Improvement
                        improvement = summary['improvement']
                        improvement_color = get_color_class(improvement, is_improvement=True)
                        improvement_arrow = get_arrow(improvement)
                        improvement_text = "Better" if improvement > 0 else "Worse" if improvement < 0 else "Same"

                        with col3:
                            st.markdown(f"""
                            <div style="padding: 1rem; background-color: rgba(26, 35, 50, 0.5); border-radius: 0.5rem;">
                                <div style="font-size: 0.875rem; color: #f5f5f5; margin-bottom: 0.5rem;">Strategy Improvement</div>
                                <div style="font-size: 2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.25rem;">{improvement:.1f}%</div>
                                <div class="{improvement_color}" style="font-size: 0.875rem; font-weight: bold;">
                                    {improvement_arrow} {improvement_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Add CSS for custom metric colors
                        st.markdown("""
                        <style>
                            .metric-positive-green {
                                color: #00cc88 !important;
                            }
                            .metric-negative-red {
                                color: #ff4b4b !important;
                            }
                            .metric-positive-red {
                                color: #ff4b4b !important;
                            }
                            .metric-negative-green {
                                color: #00cc88 !important;
                            }
                        </style>
                        """, unsafe_allow_html=True)

                        # Plot comprehensive charts
                        st.markdown("---")
                        st.subheader("📊 Comprehensive Analysis Charts")

                        # Set matplotlib font
                        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
                        plt.rcParams['axes.unicode_minus'] = False

                        # Create figure with multiple subplots
                        fig = plt.figure(figsize=(16, 12))
                        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

                        # Subplot 1: Cumulative Returns Comparison (main chart)
                        ax1 = fig.add_subplot(gs[0, :])
                        buyhold_pct = df['Buy-and-Hold Cumulative'] * 100
                        strategy_pct = df['Strategy Cumulative'] * 100
                        ax1.plot(df.index, buyhold_pct, label='Buy-and-Hold', linewidth=2.5, color='#4A90E2', alpha=0.8)
                        ax1.plot(df.index, strategy_pct, label='Livermore Strategy', linewidth=2.5, color='#F5A623',
                                 alpha=0.8)
                        ax1.set_title(f"{final_stock}: Cumulative Returns Comparison", fontsize=14, fontweight='bold',
                                      color='#d4af37')
                        ax1.set_xlabel("Date", fontsize=11)
                        ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
                        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
                        ax1.grid(True, alpha=0.3, linestyle='--')
                        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
                        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

                        # Subplot 2: Stock Price with Moving Averages
                        ax2 = fig.add_subplot(gs[1, 0])
                        ax2.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='#2E7D32', alpha=0.7)
                        if '50MA' in df.columns:
                            ax2.plot(df.index, df['50MA'], label='50-Day MA', linewidth=1.5, color='#FF6F00', alpha=0.7,
                                     linestyle='--')
                        if '200MA' in df.columns:
                            ax2.plot(df.index, df['200MA'], label='200-Day MA', linewidth=1.5, color='#C62828',
                                     alpha=0.7, linestyle='--')
                        # Mark buy/sell signals
                        signals = summary.get('signals', pd.DataFrame())
                        if not signals.empty:
                            buy_signals = signals[signals['Position'] == 1]
                            sell_signals = signals[signals['Position'] == 0]
                            if not buy_signals.empty:
                                ax2.scatter(buy_signals.index, buy_signals['Close'],
                                            color='green', marker='^', s=100, label='Buy Signal', zorder=5, alpha=0.8)
                            if not sell_signals.empty:
                                ax2.scatter(sell_signals.index, sell_signals['Close'],
                                            color='red', marker='v', s=100, label='Sell Signal', zorder=5, alpha=0.8)
                        ax2.set_title("Stock Price & Moving Averages", fontsize=12, fontweight='bold', color='#d4af37')
                        ax2.set_xlabel("Date", fontsize=10)
                        ax2.set_ylabel("Price ($)", fontsize=10)
                        ax2.legend(loc='best', fontsize=8, framealpha=0.9)
                        ax2.grid(True, alpha=0.3, linestyle='--')
                        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

                        # Subplot 3: Daily Returns Distribution
                        ax3 = fig.add_subplot(gs[1, 1])
                        buyhold_daily = df['Buy-and-Hold Return'].dropna() * 100
                        strategy_daily = df['Strategy Return'].dropna() * 100
                        ax3.hist(buyhold_daily, bins=50, alpha=0.6, label='Buy-and-Hold', color='#4A90E2',
                                 edgecolor='black')
                        ax3.hist(strategy_daily, bins=50, alpha=0.6, label='Livermore Strategy', color='#F5A623',
                                 edgecolor='black')
                        ax3.set_title("Daily Returns Distribution", fontsize=12, fontweight='bold', color='#d4af37')
                        ax3.set_xlabel("Daily Return (%)", fontsize=10)
                        ax3.set_ylabel("Frequency", fontsize=10)
                        ax3.legend(loc='best', fontsize=9, framealpha=0.9)
                        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
                        ax3.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

                        # Subplot 4: Rolling Volatility (30-day)
                        ax4 = fig.add_subplot(gs[2, 0])
                        buyhold_vol = df['Buy-and-Hold Return'].rolling(window=30).std() * 100 * np.sqrt(252)
                        strategy_vol = df['Strategy Return'].rolling(window=30).std() * 100 * np.sqrt(252)
                        ax4.plot(df.index, buyhold_vol, label='Buy-and-Hold Volatility', linewidth=1.5, color='#4A90E2',
                                 alpha=0.7)
                        ax4.plot(df.index, strategy_vol, label='Strategy Volatility', linewidth=1.5, color='#F5A623',
                                 alpha=0.7)
                        ax4.set_title("30-Day Rolling Volatility (Annualized)", fontsize=12, fontweight='bold',
                                      color='#d4af37')
                        ax4.set_xlabel("Date", fontsize=10)
                        ax4.set_ylabel("Volatility (%)", fontsize=10)
                        ax4.legend(loc='best', fontsize=9, framealpha=0.9)
                        ax4.grid(True, alpha=0.3, linestyle='--')
                        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

                        # Subplot 5: Drawdown Analysis
                        ax5 = fig.add_subplot(gs[2, 1])
                        buyhold_cum = df['Buy-and-Hold Cumulative']
                        strategy_cum = df['Strategy Cumulative']
                        buyhold_dd = (buyhold_cum - buyhold_cum.expanding().max()) * 100
                        strategy_dd = (strategy_cum - strategy_cum.expanding().max()) * 100
                        ax5.fill_between(df.index, buyhold_dd, 0, alpha=0.5, label='Buy-and-Hold Drawdown',
                                         color='#4A90E2')
                        ax5.fill_between(df.index, strategy_dd, 0, alpha=0.5, label='Strategy Drawdown',
                                         color='#F5A623')
                        ax5.set_title("Drawdown Analysis", fontsize=12, fontweight='bold', color='#d4af37')
                        ax5.set_xlabel("Date", fontsize=10)
                        ax5.set_ylabel("Drawdown (%)", fontsize=10)
                        ax5.legend(loc='best', fontsize=9, framealpha=0.9)
                        ax5.grid(True, alpha=0.3, linestyle='--')
                        ax5.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
                        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

                        # Set figure background color to match theme
                        fig.patch.set_facecolor('#1a2332')
                        for ax in [ax1, ax2, ax3, ax4, ax5]:
                            ax.set_facecolor('#1a2332')
                            ax.tick_params(colors='#f5f5f5')
                            ax.spines['bottom'].set_color('#d4af37')
                            ax.spines['top'].set_color('#d4af37')
                            ax.spines['right'].set_color('#d4af37')
                            ax.spines['left'].set_color('#d4af37')
                            ax.xaxis.label.set_color('#f5f5f5')
                            ax.yaxis.label.set_color('#f5f5f5')
                            ax.title.set_color('#d4af37')

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Strategy explanation
                        st.markdown("---")
                        st.subheader("💡 Livermore Strategy Explanation")
                        st.markdown("""
                        **Core Principles:**
                        - **Breakout Entry**: Buy when price breaks above the 20-day high
                        - **Trend Confirmation**: Only trade when price is above both 50-day and 200-day moving averages
                        - **Stop Loss Exit**: Sell when price drops below the 20-day low
                        - **Trend Following**: Only hold positions when trend direction is clear, avoid counter-trend trading
                        """)

                        # Display recent trading signals
                        st.markdown("---")
                        st.subheader("📋 Recent Trading Signals")

                        signals = summary['signals']
                        if not signals.empty:
                            signal_df = pd.DataFrame({
                                'Date': signals.index,
                                'Price': signals['Close'].values,
                                'Signal': signals['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell'),
                                '50-Day MA': signals['50MA'].values,
                                '200-Day MA': signals['200MA'].values
                            })
                            st.dataframe(signal_df.tail(10), use_container_width=True)
                        else:
                            st.info("No trading signals in the selected time range")

        else:
            st.markdown("---")
            if stock_error:
                st.warning("⚠️ Please correct the stock code input error first")
            else:
                st.info("👆 Click the button above to start backtest analysis")


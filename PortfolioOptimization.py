# === STREAMLIT DASHBOARD FOR DOW JONES STOCK ANALYSIS ===

# === IMPORT LIBRARIES ===
import streamlit as st                  # Web app framework for interactive dashboards
import yfinance as yf                  # Fetching stock data from Yahoo Finance
import pandas as pd                    # Data analysis and manipulation
from datetime import datetime, date, timedelta  # Date and time utilities
import numpy as np                     # Numerical operations
from scipy.optimize import minimize    # Optimization for portfolio allocation
import matplotlib.pyplot as plt        # Plotting for charts and heatmaps

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# === DICTIONARY: Ticker Symbols and Company Names ===
ticker_names = {
    # Dow Jones 30 stock tickers mapped to their company names
    "AAPL": "Apple Inc.", "AMGN": "Amgen Inc.", "AXP": "American Express", "BA": "Boeing Co.",
    "CAT": "Caterpillar Inc.", "CSCO": "Cisco Systems", "CVX": "Chevron Corp.", "DIS": "Walt Disney Co.",
    "DOW": "Dow Inc.", "GS": "Goldman Sachs", "HD": "Home Depot", "HON": "Honeywell",
    "IBM": "IBM Corp.", "INTC": "Intel Corp.", "JNJ": "Johnson & Johnson", "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola Co.", "MCD": "McDonald's", "MMM": "3M Company", "MRK": "Merck & Co.",
    "MSFT": "Microsoft Corp.", "NKE": "Nike Inc.", "PG": "Procter & Gamble", "TRV": "Travelers Companies",
    "UNH": "UnitedHealth Group", "V": "Visa Inc.", "VZ": "Verizon Communications",
    "WBA": "Walgreens Boots Alliance", "WMT": "Walmart Inc.", "RTX": "RTX Corp."
}

# List and dictionary to convert display names like "Apple Inc. (AAPL)" into tickers and back
display_names = [f"{name} ({ticker})" for ticker, name in ticker_names.items()]
ticker_lookup = {f"{name} ({ticker})": ticker for ticker, name in ticker_names.items()}

# === SIDEBAR FOR USER INPUTS ===
st.sidebar.image("Logo.png", width=150)

# --- Date selection for historical analysis ---
st.sidebar.subheader("Date Range")
default_end = date.today()
default_start = default_end - timedelta(days=365)
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

# --- Select stocks to analyze ---
selected_display_names = st.sidebar.multiselect("Select Stocks", display_names, default=display_names[:5])
selected_tickers = [ticker_lookup[name] for name in selected_display_names]

# --- Investment amount input ---
st.sidebar.subheader("Investment Amount")
investment_amount = st.sidebar.number_input("Total investment (€)", min_value=1000, max_value=1_000_000, value=10000)

# --- Optimization objective ---
st.sidebar.subheader("Optimization Strategy")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Data will be refreshed.")

target_type = st.sidebar.radio("Objective:", ["Max Sharpe Ratio", "Min Volatility", "Target Return"])
target_return = None
if target_type == "Target Return":
    target_return = st.sidebar.slider("Target return (%)", 0.0, 25.0, 10.0) / 100  # Convert % to decimal

# === FUNCTION: Fetch current price and volume data ===
@st.cache_data(ttl=60)  # Cached for 60 seconds to reduce API calls
def fetch_price_volume(tickers):
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            data.append({
                'Ticker': ticker,
                'Name': info.get('shortName'),
                'Sector': info.get('sector'),
                'Price': info.get('regularMarketPrice'),
                'Volume': info.get('regularMarketVolume'),
                'P/E Ratio': info.get('trailingPE'),
                'Market Cap': info.get('marketCap'),
                'Change (%)': info.get('regularMarketChangePercent')
            })
        except:
            data.append({'Ticker': ticker, 'Price': None, 'Volume': None})
    return pd.DataFrame(data)

# === TAB STRUCTURE ===
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",                # Tab 1: Real-time overview of stock data
    "Price Evolution",         # Tab 2: Historical prices and analysis
    "Portfolio Optimization",  # Tab 3: Optimization strategies for allocations
    "Metrics"                  # Tab 4: Performance metrics of the optimized portfolio
])

# === TAB 1: OVERVIEW ===
# Description: Shows current price, volume, market cap, sector, and daily performance
with tab1:
    st.title("Stock Dashboard - Dow Jones 30")
    if selected_tickers:
        df = fetch_price_volume(selected_tickers)
        ticker_strip = ""

        # Ticker strip showing price change % and direction (🔺/🔻)
        for _, row in df.iterrows():
            change = row['Change (%)']
            sign = "🔺" if change and change > 0 else "🔻"
            color = "green" if change and change > 0 else "red"
            ticker_strip += f"<span style='margin-right:20px; color:{color}'><b>{row['Ticker']}</b>: {row['Price']} ({sign} {change:.2f}%)</span>"
        st.markdown(f"<div style='white-space: nowrap; overflow-x: auto;'>{ticker_strip}</div>", unsafe_allow_html=True)

        # Data table of all selected stocks
        display_df = df.set_index("Ticker")
        display_df["Market Cap"] = display_df["Market Cap"].apply(lambda x: f"{x:,}" if pd.notnull(x) else "N/A")
        st.dataframe(display_df, use_container_width=True)
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === TAB 2: PRICE EVOLUTION ===
# Description: Explore historical trends including normalization, drawdowns, volatility, etc.
with tab2:
    st.subheader("Price Evolution")
    if selected_tickers:
        try:
            # Frequency selector for resampling
            freq = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"], horizontal=True)
            resample_map = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M"}

            # Download and resample stock prices
            raw_data = yf.download(selected_tickers, start=start_date, end=end_date)
            hist_prices = raw_data["Close"].resample(resample_map[freq]).last().dropna()

            # Normalized price chart (rebased to 100)
            norm_prices = hist_prices / hist_prices.iloc[0] * 100
            with st.expander("Rebased Price Chart", expanded=True):
                st.line_chart(norm_prices)

            # Drawdowns: percentage drops from previous peaks
            with st.expander("Drawdowns"):
                drawdowns = (norm_prices / norm_prices.cummax()) - 1
                st.line_chart(drawdowns)

            # Identify best performer
            with st.expander("Top Performer"):
                final_vals = norm_prices.iloc[-1]
                best = final_vals.idxmax()
                st.success(f"Top Performer: **{best}** — {final_vals[best]:.2f}")

            # Volatility: 30-period rolling std deviation
            with st.expander("30-Period Rolling Volatility"):
                vol = hist_prices.pct_change().rolling(30).std()
                st.line_chart(vol)

            # Moving Averages: 20-day and 50-day
            with st.expander("20d / 50d Moving Averages"):
                for ticker in selected_tickers:
                    ma20 = hist_prices[ticker].rolling(20).mean()
                    ma50 = hist_prices[ticker].rolling(50).mean()
                    ma_df = pd.DataFrame({
                        f"{ticker}": hist_prices[ticker],
                        "MA20": ma20,
                        "MA50": ma50
                    })
                    st.line_chart(ma_df, use_container_width=True)

            # Correlation heatmap between stock returns
            with st.expander("Correlation Between Stocks"):
                returns_corr = hist_prices.pct_change().dropna().corr()
                fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1e1e1e')
                fig.patch.set_facecolor('#1e1e1e')
                ax.set_facecolor('#1e1e1e')
                cax = ax.matshow(returns_corr, cmap="coolwarm")
                cb = fig.colorbar(cax)
                cb.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
                ax.set_xticks(np.arange(len(returns_corr.columns)))
                ax.set_yticks(np.arange(len(returns_corr.columns)))
                ax.set_xticklabels(returns_corr.columns, color='white', rotation=45, ha='left')
                ax.set_yticklabels(returns_corr.columns, color='white')
                for i in range(len(returns_corr.columns)):
                    for j in range(len(returns_corr.columns)):
                        ax.text(j, i, f"{returns_corr.iloc[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)
                st.pyplot(fig)

            # Year-to-date (YTD) return bar chart
            with st.expander("YTD Return"):
                ytd_return = (hist_prices.iloc[-1] / hist_prices.iloc[0]) - 1
                st.bar_chart(ytd_return)

        except Exception as e:
            st.warning(f"Could not load historical prices: {e}")

# === TAB 3: PORTFOLIO OPTIMIZATION ===
with tab3:
    st.subheader("Portfolio Optimization")

    # Ensure the user has selected at least 2 stocks for portfolio construction
    if len(selected_tickers) >= 2:
        if st.button("Run Optimization"):
            try:
                # Download historical price data
                prices = yf.download(selected_tickers, start=start_date, end=end_date)["Close"]
                returns = prices.pct_change().dropna()

                # Calculate annualized mean returns and covariance matrix
                mean_returns = returns.mean() * 252          # 252 trading days in a year
                cov_matrix = returns.cov() * 252

                num_assets = len(selected_tickers)
                init_guess = [1. / num_assets] * num_assets  # Equal weights as initial guess
                bounds = [(0.0, 1.0) for _ in range(num_assets)]  # Weights must be between 0 and 1
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights must sum to 1

                # Define objective function based on user strategy
                if target_type == "Max Sharpe Ratio":
                    def objective(weights):
                        ret = np.dot(weights, mean_returns)
                        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        return -ret / vol  # Maximize Sharpe Ratio (by minimizing its negative)
                elif target_type == "Min Volatility":
                    def objective(weights):
                        return np.dot(weights.T, np.dot(cov_matrix, weights))  # Minimize variance
                elif target_type == "Target Return":
                    def objective(weights):
                        return np.dot(weights.T, np.dot(cov_matrix, weights))  # Minimize risk
                    constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return})

                # Run optimization using Sequential Least Squares Programming (SLSQP)
                result = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)

                if result.success:
                    # Extract results
                    weights = result.x
                    weight_dict = dict(zip(selected_tickers, np.round(weights, 4)))
                    filtered_weights = {k: v for k, v in weight_dict.items() if v > 0}  # Remove 0-weights

                    # Display allocation table
                    weights_df = pd.DataFrame.from_dict(filtered_weights, orient='index', columns=['Weight'])
                    weights_df.index.name = 'Ticker'
                    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")

                    # Show amount allocated per stock
                    allocations = {ticker: weight * investment_amount for ticker, weight in weight_dict.items() if weight > 0}
                    alloc_df = pd.DataFrame.from_dict(allocations, orient='index', columns=['Allocated (€)'])
                    alloc_df['Allocated (€)'] = alloc_df['Allocated (€)'].apply(lambda x: f"{x:,.2f} €")
                    alloc_df.index.name = 'Ticker'

                    # Combine and display both tables
                    combined_df = weights_df.join(alloc_df)
                    st.table(combined_df)

                    # Pie chart of allocations
                    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1e1e1e')
                    ax.pie(filtered_weights.values(), labels=filtered_weights.keys(), autopct='%1.1f%%',
                           startangle=140, textprops={'color': 'white', 'fontsize': 8},
                           wedgeprops=dict(width=0.4))
                    ax.axis('equal')  # Equal aspect ratio to ensure circle
                    st.pyplot(fig)

                    # Portfolio Performance Metrics
                    port_return = np.dot(weights, mean_returns)
                    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe = port_return / port_vol

                    # Save metrics in session state to use in Tab 4
                    metrics_df = pd.DataFrame({
                        "Metric": ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio"],
                        "Value": [f"{port_return:.2%}", f"{port_vol:.2%}", f"{sharpe:.2f}"]
                    }).set_index("Metric")
                    st.session_state.metrics_df = metrics_df

                    # Portfolio Value Over Time (Cumulative)
                    portfolio_returns = (returns @ weights).dropna()
                    cumulative_value = (1 + portfolio_returns).cumprod() * investment_amount
                    st.subheader("Cumulative Portfolio Value Over Time")
                    st.line_chart(cumulative_value)

                    # Benchmark Comparison — Optimized vs S&P 500
                    # Benchmark Comparison — Optimized vs S&P 500
                    try:
                        sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
                        sp500_returns = sp500.pct_change().dropna()
                        sp500_cumulative = (1 + sp500_returns).cumprod() * investment_amount
                        sp500_cumulative = sp500_cumulative.squeeze()

                        comparison_df = pd.DataFrame({
                            "Optimized Portfolio (€)": cumulative_value.squeeze(),  # Also ensure this is 1D
                            "S&P 500 (€)": sp500_cumulative
                        })
                        st.subheader("Portfolio vs S&P 500")
                        st.line_chart(comparison_df)

                    except Exception as e:
                        st.warning(f"Couldn't load S&P 500 benchmark data: {e}")

                    # Comparison with Equal Weight Portfolio
                    equal_weights = np.array([1 / num_assets] * num_assets)
                    equal_returns = (returns @ equal_weights).dropna()
                    equal_cumulative = (1 + equal_returns).cumprod() * investment_amount

                    comparison_df = pd.DataFrame({
                        "Optimized Portfolio (€)": cumulative_value,
                        "Equal Weight Portfolio (€)": equal_cumulative
                    })
                    st.subheader("Equal Weight vs Optimized Portfolio")
                    st.line_chart(comparison_df)

                else:
                    st.error("Optimization failed.")  # If solver didn't converge

            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        st.info("Please select at least 2 stocks.")  # Warning if not enough assets

# === TAB 4: METRICS ===
with tab4:
    st.subheader("Portfolio Performance Metrics")

    # Display metrics only if they have been computed in Tab 3
    if 'metrics_df' in st.session_state:
        st.table(st.session_state.metrics_df)
    else:
        st.info("Run optimization first to see portfolio metrics.")  # Prompt user to go to Tab 3


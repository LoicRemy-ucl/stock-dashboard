# === FILE: app.py ===
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from src.optimizer import PortfolioOptimizer
from src.dashboard import StockDashboard
from src import visuals

# --- INIT ---

dashboard = StockDashboard(PortfolioOptimizer)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.sidebar.image("Logo.png", width=150)

# --- SETTINGS ---
st.sidebar.subheader("Enter Stock or ETF Tickers")
user_input = st.sidebar.text_input("Comma-separated tickers (e.g., AAPL,MSFT,TSLA,VOO)", value="AAPL,MSFT,TSLA")

selected_tickers = [ticker.strip().upper() for ticker in user_input.split(",") if ticker.strip()]

if not selected_tickers:
    st.sidebar.warning("Please enter at least one valid ticker symbol.")


# --- SIDEBAR ---
st.sidebar.subheader("Date Range")
default_end = date.today()
default_start = default_end - timedelta(days=365)
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

st.sidebar.subheader("Investment Amount")
investment_amount = st.sidebar.number_input("Total investment (€)", min_value=1000, max_value=1_000_000, value=10000)

st.sidebar.subheader("Optimization Strategy")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared.")

target_type = st.sidebar.radio("Objective:", ["Max Sharpe Ratio", "Min Volatility", "Target Return"])
target_return = None
if target_type == "Target Return":
    target_return = st.sidebar.slider("Target return (%)", 0.0, 25.0, 10.0) / 100

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Price Evolution", "Portfolio Optimization", "Metrics"])

# === TAB 1: OVERVIEW ===
with tab1:
    st.title("Stock Dashboard")
    if selected_tickers:
        df = dashboard.get_current_overview(selected_tickers)
        ticker_strip = ""
        for _, row in df.iterrows():
            change = row['Change (%)']
            sign = "🔺" if change and change > 0 else "🔻"
            color = "green" if change and change > 0 else "red"
            ticker_strip += f"<span style='margin-right:20px; color:{color}'><b>{row['Ticker']}</b>: {row['Price']} ({sign} {change:.2f}%)</span>"
        st.markdown(f"<div style='white-space: nowrap; overflow-x: auto;'>{ticker_strip}</div>", unsafe_allow_html=True)

        df["Market Cap"] = df["Market Cap"].apply(lambda x: f"{x:,}" if pd.notnull(x) else "N/A")
        st.dataframe(df.set_index("Ticker"), use_container_width=True)
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === TAB 2: PRICE EVOLUTION ===
with tab2:
    st.subheader("Price Evolution")
    if selected_tickers:
        try:
            freq = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"], horizontal=True)
            hist_prices = dashboard.get_historical_data(selected_tickers, start_date, end_date, freq)
            norm_prices = hist_prices / hist_prices.iloc[0] * 100

            with st.expander("Rebased Price Chart", expanded=True):
                visuals.plot_normalized_prices(norm_prices)

            with st.expander("Drawdowns"):
                visuals.plot_drawdowns(norm_prices)

            with st.expander("Top Performer"):
                final_vals = norm_prices.iloc[-1]
                best = final_vals.idxmax()
                st.success(f"Top Performer: **{best}** — {final_vals[best]:.2f}")

            with st.expander("30-Period Rolling Volatility"):
                visuals.plot_rolling_volatility(hist_prices)

            with st.expander("20d / 50d Moving Averages"):
                visuals.plot_moving_averages(hist_prices, selected_tickers)

            with st.expander("Correlation Between Stocks"):
                returns_corr = hist_prices.pct_change().dropna().corr()
                visuals.plot_correlation_heatmap(returns_corr)

            with st.expander("YTD Return"):
                ytd_return = (hist_prices.iloc[-1] / hist_prices.iloc[0]) - 1
                visuals.plot_ytd_return(ytd_return)

        except Exception as e:
            st.warning(f"Could not load historical prices: {e}")

# === TAB 3: OPTIMIZATION ===
with tab3:
    st.subheader("Portfolio Optimization")
    if len(selected_tickers) >= 2 and st.button("Run Optimization"):
        try:
            returns, filtered_weights, allocations, mean_returns, cov_matrix, weights, ignored_tickers = dashboard.run_optimizer(
                selected_tickers, start_date, end_date, target_type, investment_amount, target_return
            )
            if ignored_tickers:
                st.warning(f"The following tickers were ignored due to missing data: {', '.join(ignored_tickers)}")
            # Display weights and allocations
            # Show all tickers, even those with 0% weight
            weight_dict = dict(zip(selected_tickers, weights))  # Use this to ensure all are included
            weights_df = pd.DataFrame.from_dict(weight_dict, orient='index', columns=['Weight'])
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")

            alloc_dict = {ticker: weight * investment_amount for ticker, weight in weight_dict.items()}
            alloc_df = pd.DataFrame.from_dict(alloc_dict, orient='index', columns=['Allocated (€)'])
            alloc_df['Allocated (€)'] = alloc_df['Allocated (€)'].apply(lambda x: f"{x:,.2f} €")

            combined_df = weights_df.join(alloc_df)
            st.table(combined_df)

            # Pie chart
            visuals.plot_allocation_pie(filtered_weights)

            # Portfolio metrics
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = port_return / port_vol
            metrics_df = pd.DataFrame({
                "Metric": ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio"],
                "Value": [f"{port_return:.2%}", f"{port_vol:.2%}", f"{sharpe:.2f}"]
            }).set_index("Metric")
            st.session_state.metrics_df = metrics_df

            # Portfolio performance over time
            cumulative = dashboard.compute_cumulative_returns(returns, weights, investment_amount)
            st.subheader("Cumulative Portfolio Value Over Time")
            st.line_chart(cumulative)

            comparison = dashboard.compute_SP500(returns, weights, investment_amount, start_date, end_date)
            st.subheader("Portfolio vs SP500")
            st.line_chart(comparison)

            comparison_equal = dashboard.compute_equally_weighted(returns, weights, investment_amount)
            st.subheader("Portfolio vs Equal Weight")
            st.line_chart(comparison_equal)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.info("Please select at least 2 stocks.")

# === TAB 4: METRICS ===
with tab4:
    st.subheader("Portfolio Performance Metrics")
    if 'metrics_df' in st.session_state:
        st.table(st.session_state.metrics_df)
    else:
        st.info("Run optimization first to see portfolio metrics.")

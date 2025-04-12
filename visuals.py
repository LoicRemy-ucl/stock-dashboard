import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd


def plot_normalized_prices(norm_prices):
    st.line_chart(norm_prices)


def plot_drawdowns(norm_prices):
    drawdowns = (norm_prices / norm_prices.cummax()) - 1
    st.line_chart(drawdowns)


def plot_rolling_volatility(hist_prices):
    vol = hist_prices.pct_change().rolling(30).std()
    st.line_chart(vol)


def plot_moving_averages(hist_prices, tickers):
    for ticker in tickers:
        ma20 = hist_prices[ticker].rolling(20).mean()
        ma50 = hist_prices[ticker].rolling(50).mean()
        ma_df = pd.DataFrame({
            f"{ticker}": hist_prices[ticker],
            "MA20": ma20,
            "MA50": ma50
        })
        st.line_chart(ma_df, use_container_width=True)


def plot_correlation_heatmap(returns_corr):
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


def plot_ytd_return(ytd_return):
    st.bar_chart(ytd_return)


def plot_portfolio_comparison(comparison_df):
    st.line_chart(comparison_df)


def plot_allocation_pie(filtered_weights):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1e1e1e')
    ax.pie(filtered_weights.values(), labels=filtered_weights.keys(), autopct='%1.1f%%',
           startangle=140, textprops={'color': 'white', 'fontsize': 8},
           wedgeprops=dict(width=0.4))
    ax.axis('equal')
    st.pyplot(fig)

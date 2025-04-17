from src.data import StockDataFetcher
from src.optimizer import PortfolioOptimizer
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from src import visuals


class StockDashboard:
    def __init__(self, optimizer):
        self.fetcher = StockDataFetcher()

    def get_current_overview(self, tickers):
        return self.fetcher.fetch_price_volume(tickers)

    def get_historical_data(self, tickers, start_date, end_date, frequency):
        resample_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
        return self.fetcher.download_price_data(tickers, start_date, end_date, resample_map[frequency])

    def run_optimizer(self, tickers, start_date, end_date, target_type, investment_amount, target_return=None):
        prices = self.fetcher.download_price_data(tickers, start_date, end_date)

        # Drop columns with all NaNs (invalid tickers)
        valid_prices = prices.dropna(axis=1, how='all')
        valid_tickers = valid_prices.columns.tolist()
        ignored_tickers = list(set(tickers) - set(valid_tickers))

        if len(valid_tickers) < 2:
            raise ValueError("Need at least 2 valid tickers with historical data for optimization.")

        returns = valid_prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        optimizer = PortfolioOptimizer(returns, mean_returns, cov_matrix, target_type, target_return)
        weights = optimizer.optimize()

        weight_dict = dict(zip(valid_tickers, weights))
        filtered_weights = {k: v for k, v in weight_dict.items() if v > 0}
        allocations = {ticker: weight * investment_amount for ticker, weight in weight_dict.items() if weight > 0}

        return returns, filtered_weights, allocations, mean_returns, cov_matrix, weights, ignored_tickers

    def compute_cumulative_returns(self, returns, weights, investment_amount):
        portfolio_returns = (returns @ weights).dropna()
        cumulative_value = (1 + portfolio_returns).cumprod() * investment_amount
        return cumulative_value

    def compute_SP500(self, returns, weights, investment_amount, start_date, end_date):
        cumulative_value = self.compute_cumulative_returns(returns, weights, investment_amount)
        sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
        sp500_returns = sp500.pct_change().dropna()
        sp500_cumulative = (1 + sp500_returns).cumprod() * investment_amount
        sp500_cumulative = sp500_cumulative.squeeze()
        comparison_df = pd.DataFrame({
            "Optimized Portfolio (€)": cumulative_value.squeeze(),  # Also ensure this is 1D
            "S&P 500 (€)": sp500_cumulative
        })
        return comparison_df

    def compute_equally_weighted(self, returns, weights, investment_amount):
        cumulative_value = self.compute_cumulative_returns(returns, weights, investment_amount)
        num_assets = len(weights)
        equal_weights = np.array([1 / num_assets] * num_assets)
        equal_returns = (returns @ equal_weights).dropna()
        equal_cumulative = (1 + equal_returns).cumprod() * investment_amount

        comparison_df = pd.DataFrame({
                "Optimized Portfolio (€)": cumulative_value,
                "Equal Weight Portfolio (€)": equal_cumulative
            })
        return comparison_df





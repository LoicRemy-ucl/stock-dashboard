import yfinance as yf
import pandas as pd


class StockDataFetcher:
    def fetch_price_volume(self, tickers):
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

    def download_price_data(self, tickers, start, end, interval="1d"):
        return yf.download(tickers, start=start, end=end, interval=interval)["Close"]

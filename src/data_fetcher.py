from contextlib import suppress
from datetime import datetime, timedelta
from os import getenv

import streamlit as st
import yfinance as yf
from fredapi import Fred


def get_fred_api_key():
    api_key = getenv("FRED_API_KEY")
    if not api_key:
        with suppress(AttributeError, RuntimeError):
            api_key = st.secrets.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY not found in environment variables or Streamlit secrets.")
    return api_key


fred = Fred(api_key=get_fred_api_key())


def get_risk_free_rate(maturity_years):
    series_map = {
        1 / 12: "DTB4WK",
        2 / 12: "DTB4WK",
        3 / 12: "DTB3",
        6 / 12: "DTB6",
        1: "DGS1",
        2: "DGS2",
        3: "DGS3",
        5: "DGS5",
        7: "DGS7",
        10: "DGS10",
        20: "DGS20",
        30: "DGS30",
    }
    closest_maturity = min(series_map.keys(), key=lambda x: abs(x - maturity_years))
    series_id = series_map[closest_maturity]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    try:
        data = fred.get_series(series_id, start_date, end_date)
        if not data.empty:
            return float(data.iloc[-1]) / 100
        else:
            return 0.05
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.05


def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1mo")

        company_name = info.get("longName", ticker)
        if not company_name or company_name == ticker:
            company_name = info.get("shortName", ticker)

        dividend_yield = 0.0
        if "dividendYield" in info and info["dividendYield"] is not None:
            dividend_yield = float(info["dividendYield"])  # yfinance returns as decimal

        current_price = 0.0
        if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
            current_price = float(info["regularMarketPrice"])
        elif not history.empty:
            current_price = float(history["Close"].iloc[-1])

        volatility = 0.0
        if not history.empty and len(history) > 1:
            volatility = float(history["Close"].pct_change().dropna().std() * (252**0.5))  # Returns as decimal

        return {
            "current_price": current_price,
            "volatility": volatility,
            "dividend_yield": dividend_yield,  # Decimal (e.g., 0.02 for 2%)
            "company_name": str(company_name),
        }

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return {
            "current_price": 0.0,
            "volatility": 0.0,
            "dividend_yield": 0.0,  # Decimal
            "company_name": str(ticker),
        }

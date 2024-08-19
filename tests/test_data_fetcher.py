import pytest
from data_fetcher import get_risk_free_rate, fetch_stock_data
import pandas as pd

def test_get_risk_free_rate():
    rate = get_risk_free_rate(1)
    assert 0 <= rate <= 0.1, f"Risk-free rate {rate} is out of expected range"

@pytest.fixture
def mock_yfinance(monkeypatch):
    class MockTicker:
        @property
        def info(self):
            return {
                'regularMarketPrice': 150.0,
                'dividendYield': 0.02,
                'longName': 'Test Company'
            }
        
        def history(self, period):
            return pd.DataFrame({
                'Close': [149, 150, 151]
            })

    def mock_ticker(*args, **kwargs):
        return MockTicker()

    monkeypatch.setattr("yfinance.Ticker", mock_ticker)

def test_fetch_stock_data_success(mock_yfinance):
    result = fetch_stock_data('TEST')
    assert result['current_price'] == 150.0
    assert result['dividend_yield'] == 0.02
    assert result['company_name'] == 'Test Company'
    assert isinstance(result['volatility'], float)
    assert 0 < result['volatility'] < 1

def test_fetch_stock_data_failure(monkeypatch):
    def mock_ticker_error(*args, **kwargs):
        raise Exception("API Error")
    
    monkeypatch.setattr("yfinance.Ticker", mock_ticker_error)
    
    result = fetch_stock_data('TEST')
    assert result == {
        "current_price": 0,
        "volatility": 0,
        "dividend_yield": 0,
        "company_name": 'TEST',
    }
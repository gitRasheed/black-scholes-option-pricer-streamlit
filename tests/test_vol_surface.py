import pandas as pd
import pytest
from plotly.graph_objs import Figure

# Fixed import to match your project's file name
from src.vol_surface import (
    calculate_implied_volatility,
    create_volatility_surface,
    fetch_option_chain,
    get_volatility_surface,
)


@pytest.fixture
def mock_option_chain():
    calls = pd.DataFrame(
        {
            "contractSymbol": ["AAPL220121C00140000", "AAPL220121C00145000"],
            "lastTradeDate": [1642784399, 1642784399],
            "strike": [140.0, 145.0],
            "lastPrice": [10.5, 6.8],
            "bid": [10.4, 6.7],
            "ask": [10.6, 6.9],
            "change": [0.5, 0.3],
            "percentChange": [5.0, 4.5],
            "volume": [2000, 1500],
            "openInterest": [5000, 4000],
            "impliedVolatility": [0.3, 0.32],
            "inTheMoney": [True, False],
            "contractSize": ["REGULAR", "REGULAR"],
            "currency": ["USD", "USD"],
            "optionType": ["call", "call"],
            "expiryDate": ["2022-01-21", "2022-01-21"],
            "timeToExpiry": [0.5, 0.5],
            "daysToExpiry": [182, 182],
            "moneyness": [1.0, 1.04],
        }
    )

    puts = pd.DataFrame(
        {
            "contractSymbol": ["AAPL220121P00140000", "AAPL220121P00145000"],
            "lastTradeDate": [1642784399, 1642784399],
            "strike": [140.0, 145.0],
            "lastPrice": [5.2, 8.3],
            "bid": [5.1, 8.2],
            "ask": [5.3, 8.4],
            "change": [-0.3, -0.2],
            "percentChange": [-5.5, -2.4],
            "volume": [1800, 1200],
            "openInterest": [4500, 3800],
            "impliedVolatility": [0.28, 0.3],
            "inTheMoney": [False, True],
            "contractSize": ["REGULAR", "REGULAR"],
            "currency": ["USD", "USD"],
            "optionType": ["put", "put"],
            "expiryDate": ["2022-01-21", "2022-01-21"],
            "timeToExpiry": [0.5, 0.5],
            "daysToExpiry": [182, 182],
            "moneyness": [1.0, 1.04],
        }
    )

    return pd.concat([calls, puts]), 140.0


@pytest.fixture
def mock_yfinance(monkeypatch, mock_option_chain):
    mock_options_df, current_price = mock_option_chain

    class MockOptionChain:
        def __init__(self):
            calls_only = mock_options_df[mock_options_df["optionType"] == "call"].copy()
            puts_only = mock_options_df[mock_options_df["optionType"] == "put"].copy()
            self.calls = calls_only.drop(
                ["optionType", "expiryDate", "timeToExpiry", "daysToExpiry", "moneyness"], axis=1
            )
            self.puts = puts_only.drop(
                ["optionType", "expiryDate", "timeToExpiry", "daysToExpiry", "moneyness"], axis=1
            )

    class MockTicker:
        @property
        def options(self):
            return ["2022-01-21", "2022-02-18"]

        def history(self, period):
            return pd.DataFrame({"Close": [current_price]})

        def option_chain(self, date):
            return MockOptionChain()

    def mock_ticker_constructor(*args, **kwargs):
        return MockTicker()

    monkeypatch.setattr("yfinance.Ticker", mock_ticker_constructor)


def test_fetch_option_chain(mock_yfinance, mock_option_chain):
    options_df, expected_price = mock_option_chain

    result_df, result_price = fetch_option_chain("AAPL")

    # Fix: Check if result_df is not None before accessing its attributes
    assert result_df is not None, "fetch_option_chain returned None DataFrame"
    assert result_price == expected_price
    assert not result_df.empty
    assert "optionType" in result_df.columns
    assert "moneyness" in result_df.columns
    assert "timeToExpiry" in result_df.columns

    assert set(result_df["optionType"].unique()) == {"call", "put"}


def test_calculate_implied_volatility():
    options_df = pd.DataFrame(
        {
            "optionType": ["call", "put"],
            "lastPrice": [5.0, 3.0],
            "strike": [100.0, 100.0],
            "timeToExpiry": [1.0, 1.0],
            "moneyness": [1.0, 1.0],
        }
    )

    result_df = calculate_implied_volatility(options_df, 0.05, 0.0)

    # Fix: Check if result_df is not None before accessing its attributes
    assert result_df is not None, "calculate_implied_volatility returned None"
    assert "impliedVolatility" in result_df.columns

    volatilities = result_df["impliedVolatility"].dropna().values
    for vol in volatilities:
        assert 0.0 <= vol <= 2.0


def test_create_volatility_surface(mock_option_chain):
    options_df, _ = mock_option_chain

    call_fig = create_volatility_surface(options_df, "call", True, None)
    assert isinstance(call_fig, Figure)

    # Fix: Access title safely using dict conversion
    call_fig_dict = call_fig.to_dict()
    assert "Call Option" in call_fig_dict.get("layout", {}).get("title", {}).get("text", "")

    put_fig = create_volatility_surface(options_df, "put", True, None)
    assert isinstance(put_fig, Figure)

    # Fix: Access title safely using dict conversion
    put_fig_dict = put_fig.to_dict()
    assert "Put Option" in put_fig_dict.get("layout", {}).get("title", {}).get("text", "")


def test_create_volatility_surface_with_filters(mock_option_chain):
    options_df, _ = mock_option_chain

    fig = create_volatility_surface(options_df, "call", True, (0.9, 1.1))
    assert isinstance(fig, Figure)

    fig = create_volatility_surface(options_df, "put", False, (130, 150))
    assert isinstance(fig, Figure)


def test_get_volatility_surface(mock_yfinance):
    call_fig, put_fig, price = get_volatility_surface("AAPL", 0.05, 0.01, True, None)

    assert price == 140.0
    assert isinstance(call_fig, Figure)
    assert isinstance(put_fig, Figure)


def test_handle_empty_data(monkeypatch):
    def mock_fetch_empty(*args, **kwargs):
        return None, 0

    monkeypatch.setattr("src.vol_surface.fetch_option_chain", mock_fetch_empty)

    call_fig, put_fig, price = get_volatility_surface("INVALID", 0.05, 0.01, True, None)

    assert price == 0
    assert call_fig is None
    assert put_fig is None

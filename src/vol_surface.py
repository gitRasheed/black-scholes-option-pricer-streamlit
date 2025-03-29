import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from src.black_scholes import BlackScholes


def fetch_option_chain(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        expiration_dates = stock.options
        all_options = []

        for date in expiration_dates:
            try:
                opt = stock.option_chain(date)

                expiry_date = datetime.datetime.strptime(date, "%Y-%m-%d")
                days_to_expiry = (expiry_date - datetime.datetime.now()).days
                years_to_expiry = days_to_expiry / 365.0

                calls = opt.calls.copy()
                calls["optionType"] = "call"
                calls["expiryDate"] = date
                calls["timeToExpiry"] = years_to_expiry
                calls["daysToExpiry"] = days_to_expiry
                calls["moneyness"] = calls["strike"] / current_price
                all_options.append(calls)

                puts = opt.puts.copy()
                puts["optionType"] = "put"
                puts["expiryDate"] = date
                puts["timeToExpiry"] = years_to_expiry
                puts["daysToExpiry"] = days_to_expiry
                puts["moneyness"] = puts["strike"] / current_price
                all_options.append(puts)

            except Exception as e:
                print(f"Error fetching options for {date}: {e}")
                continue

        if all_options:
            options_df = pd.concat(all_options)
            return options_df, current_price
        else:
            return None, current_price

    except Exception as e:
        print(f"Error fetching option chain: {e}")
        return None, 0


def calculate_implied_volatility(options_df, risk_free_rate, dividend_yield):
    def calc_iv(row, current_price, risk_free_rate, dividend_yield):
        option_type = row["optionType"]
        option_price = row["lastPrice"]
        strike = row["strike"]
        time_to_expiry = row["timeToExpiry"]

        if option_price < 0.01:
            return None

        def objective_function(sigma):
            bs = BlackScholes(current_price, strike, time_to_expiry, risk_free_rate, sigma, dividend_yield)
            model_price = bs.calculate_option_price(option_type)
            return abs(model_price - option_price)

        best_iv = 0.5
        best_error = objective_function(best_iv)

        for sigma in np.linspace(0.01, 2.0, 100):
            error = objective_function(sigma)
            if error < best_error:
                best_error = error
                best_iv = sigma

        if best_error > 0.1 * option_price:
            return None

        return best_iv

    current_price = options_df["moneyness"].iloc[0] * options_df["strike"].iloc[0]
    options_df["impliedVolatility"] = options_df.apply(
        lambda row: calc_iv(row, current_price, risk_free_rate, dividend_yield), axis=1
    )

    return options_df


def create_volatility_surface(options_df, option_type, use_moneyness=True, filter_range=None):
    options_df = options_df[options_df["optionType"] == option_type].copy()
    options_df = options_df.dropna(subset=["impliedVolatility"])

    if filter_range and use_moneyness:
        options_df = options_df[
            (options_df["moneyness"] >= filter_range[0]) & (options_df["moneyness"] <= filter_range[1])
        ]
    elif filter_range and not use_moneyness:
        options_df = options_df[(options_df["strike"] >= filter_range[0]) & (options_df["strike"] <= filter_range[1])]

    if options_df.empty:
        return None

    y_col = "moneyness" if use_moneyness else "strike"
    pivot_table = options_df.pivot_table(
        values="impliedVolatility", index="daysToExpiry", columns=y_col, aggfunc="mean"
    )

    x_data = pivot_table.index.values
    y_data = pivot_table.columns.values
    z_data = pivot_table.values

    title = f"{option_type.capitalize()} Option Implied Volatility Surface"
    colorscale = "Blues" if option_type == "call" else "Reds"

    hover_text = "Moneyness: %{y:.2f}<br>" if use_moneyness else "Strike: %{y:.2f}<br>"

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_data,
                y=y_data,
                z=z_data,
                colorscale=colorscale,
                hovertemplate="Days to Expiry: %{x}<br>" + hover_text + "Implied Vol: %{z:.2f}<extra></extra>",
            )
        ]
    )

    x_title = "Days to Expiry"
    y_title = "Moneyness (Strike/Spot)" if use_moneyness else "Strike Price"
    z_title = "Implied Volatility"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_title, camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
        ),
        height=500,
        width=650,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def get_volatility_surface(ticker, risk_free_rate, dividend_yield, use_moneyness=True, filter_range=None):
    options_df, current_price = fetch_option_chain(ticker)

    if options_df is None:
        return None, None, current_price

    options_df = calculate_implied_volatility(options_df, risk_free_rate, dividend_yield)

    call_fig = create_volatility_surface(options_df, "call", use_moneyness, filter_range)
    put_fig = create_volatility_surface(options_df, "put", use_moneyness, filter_range)

    return call_fig, put_fig, current_price

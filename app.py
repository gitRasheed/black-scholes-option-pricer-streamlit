import numpy as np
import pandas as pd
import streamlit as st

from src.black_scholes import BlackScholes
from src.data_fetcher import fetch_stock_data, get_risk_free_rate
from src.visualizations import (
    create_greeks_plot,
    create_heatmap,
    create_profit_loss_chart,
)
from src.vol_surface import get_volatility_surface

st.set_page_config(layout="wide", page_title="Options Pricer", page_icon="📈")

# Sidebar
st.sidebar.markdown(
    """
    <style>
    .linkedin-button {
        display: inline-flex;
        align-items: center;
        background-color: white;
        color: #0077B5;
        padding: 10px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        transition: all 0.3s;
        border: 2px solid #0077B5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .linkedin-button:hover {
        background-color: #f3f9ff;
        color: #004d77;
        border-color: #004d77;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .linkedin-button img {
        margin-right: 10px;
    }
    </style>
    <a href="https://www.linkedin.com/in/khoshnaw" target="_blank" class="linkedin-button">
        <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"
        width="20" height="20" />
        Created by Rasheed Khoshnaw
    </a>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Stock Information")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
stock_data = fetch_stock_data(ticker)
st.sidebar.markdown(f"**Company:** {stock_data['company_name']}")
S = st.sidebar.number_input("Current Stock Price ($)", value=stock_data["current_price"], step=0.01)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.subheader("Option &  Market Parameters")
K = st.sidebar.number_input("Strike Price ($)", value=S, step=0.01)
T = st.sidebar.number_input("Time to Maturity (years)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
sigma = st.sidebar.number_input(
    "Volatility (σ)",
    value=stock_data["volatility"],
    min_value=0.01,
    max_value=2.0,
    step=0.01,
)
r = st.sidebar.number_input(
    "Risk-free Rate (%)",
    value=get_risk_free_rate(T) * 100,
    min_value=0.0,
    max_value=20.0,
    step=0.01,
    format="%.2f",
)
q = st.sidebar.number_input(
    "Dividend Yield (%)",
    value=float(stock_data["dividend_yield"]) * 100,
    min_value=0.0,
    max_value=20.0,
    step=0.01,
    format="%.2f",
)

r /= 100
q /= 100

st.sidebar.markdown("---")

st.sidebar.subheader("Trade Information")
call_purchase_price = st.sidebar.number_input("Call Option Purchase Price ($)", value=0.0, step=0.01)
put_purchase_price = st.sidebar.number_input("Put Option Purchase Price ($)", value=0.0, step=0.01)

st.sidebar.markdown("---")

st.sidebar.subheader("Heatmap Parameters")
heatmap_price_range = st.sidebar.slider("Stock Price Range (%)", min_value=10, max_value=300, value=(80, 120), step=5)
heatmap_volatility_range = st.sidebar.slider(
    "Volatility Range (%)", min_value=10, max_value=500, value=(50, 150), step=10
)

st.sidebar.markdown("---")

st.sidebar.subheader("Volatility Surface Parameters")
use_moneyness = st.sidebar.checkbox("Use Moneyness for Y-axis", value=True)
filter_range = st.sidebar.slider(
    "Moneyness Range" if use_moneyness else "Strike Range (%)",
    min_value=0.5 if use_moneyness else 50,
    max_value=2.0 if use_moneyness else 200,
    value=(0.8, 1.2) if use_moneyness else (80, 120),
    step=0.05 if use_moneyness else 5,
)

# Body
st.title("📊 Black-Scholes Option Pricer")

bs_call = BlackScholes(S, K, T, r, sigma, q)
bs_put = BlackScholes(S, K, T, r, sigma, q)
call_price = bs_call.calculate_option_price("call")
put_price = bs_put.calculate_option_price("put")
call_greeks = bs_call.calculate_greeks()
put_greeks = bs_put.calculate_greeks()

col1, col2 = st.columns(2)


def display_greeks(greeks, option_type):
    greek_symbols = {
        "delta": "Δ (Delta)",
        "gamma": "Γ (Gamma)",
        "vega": "ν (Vega)",
        "theta": "Θ (Theta)",
        "rho": "ρ (Rho)",
    }

    greek_data = {
        greek_symbols[g.split("_")[0]]: f"{v:.4f}"
        for g, v in greeks.items()
        if g.endswith(option_type) or g == "gamma" or g == "vega"
    }

    return pd.DataFrame(greek_data.items(), columns=["Greek", "Value"]).set_index("Greek")


def get_pricing_status(purchase_price, model_price, threshold=0.01):
    if purchase_price == 0:
        return ""
    difference = (purchase_price - model_price) / model_price
    if abs(difference) <= threshold:
        return "Fairly Priced"
    elif difference > threshold:
        return f"Overpriced ({difference:.2%})"
    else:
        return f"Underpriced ({-difference:.2%})"


with col1:
    st.subheader("Call Option")
    st.metric("Price", f"${call_price:.2f}", delta=None)

    st.write("Call Greeks")
    st.table(display_greeks(call_greeks, "call"))

    call_pnl = call_price - call_purchase_price
    status = get_pricing_status(call_purchase_price, call_price)

    if call_pnl > 0:
        st.success(f"Profit: ${call_pnl:.2f}   —   {status}")
    elif call_pnl < 0:
        st.error(f"Loss: ${call_pnl:.2f}   —   {status}")
    else:
        st.info(f"Break-even   —   {status}")

with col2:
    st.subheader("Put Option")
    st.metric("Price", f"${put_price:.2f}", delta=None)

    st.write("Put Greeks")
    st.table(display_greeks(put_greeks, "put"))

    put_pnl = put_price - put_purchase_price
    status = get_pricing_status(put_purchase_price, put_price)

    if put_pnl > 0:
        st.success(f"Profit: ${put_pnl:.2f}   —   {status}")
    elif put_pnl < 0:
        st.error(f"Loss: ${put_pnl:.2f}   —   {status}")
    else:
        st.info(f"Break-even   —   {status}")

st.markdown("---")

st.subheader("Option Price and PnL Heatmap")
S_range = np.linspace(S * heatmap_price_range[0] / 100, S * heatmap_price_range[1] / 100, 50)
sigma_range = np.linspace(
    sigma * heatmap_volatility_range[0] / 100,
    sigma * heatmap_volatility_range[1] / 100,
    50,
)

call_prices = np.array(
    [[BlackScholes(s, K, T, r, sig, q).calculate_option_price("call") for s in S_range] for sig in sigma_range]
)
put_prices = np.array(
    [[BlackScholes(s, K, T, r, sig, q).calculate_option_price("put") for s in S_range] for sig in sigma_range]
)

call_pnl = call_prices - call_purchase_price
put_pnl = put_prices - put_purchase_price

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_heatmap(S_range, sigma_range, call_pnl, call_prices, "Call Option PnL"))
with col2:
    st.plotly_chart(create_heatmap(S_range, sigma_range, put_pnl, put_prices, "Put Option PnL"))

st.markdown("---")

st.subheader("Profit/Loss Chart")
S_range = np.linspace(0.5 * K, 1.5 * K, 100)  # type: ignore
call_pnl_range = [max(s - K, 0) - call_purchase_price for s in S_range]
put_pnl_range = [max(K - s, 0) - put_purchase_price for s in S_range]
call_break_even = S_range[np.argmin(np.abs(call_pnl_range))]
put_break_even = S_range[np.argmin(np.abs(put_pnl_range))]
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_profit_loss_chart(S_range, call_pnl_range, call_break_even, "Call Option P&L"))
with col2:
    st.plotly_chart(create_profit_loss_chart(S_range, put_pnl_range, put_break_even, "Put Option P&L"))

st.markdown("---")

st.subheader("Greeks")
call_greeks_values = {
    greek: [BlackScholes(s, K, T, r, sigma, q).calculate_greeks()[greek] for s in S_range]
    for greek in ["delta_call", "gamma", "vega", "theta_call", "rho_call"]
}
put_greeks_values = {
    greek: [BlackScholes(s, K, T, r, sigma, q).calculate_greeks()[greek] for s in S_range]
    for greek in ["delta_put", "gamma", "vega", "theta_put", "rho_put"]
}
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_greeks_plot(S_range, call_greeks_values, "Call Option Greeks"))
with col2:
    st.plotly_chart(create_greeks_plot(S_range, put_greeks_values, "Put Option Greeks"))

st.markdown("---")

st.subheader("Implied Volatility Surface")

actual_filter_range = filter_range
if not use_moneyness:
    actual_filter_range = (S * filter_range[0] / 100, S * filter_range[1] / 100)

with st.spinner("Fetching option chain data..."):
    call_surface, put_surface, _ = get_volatility_surface(ticker, r, q, use_moneyness, actual_filter_range)

col1, col2 = st.columns(2)
with col1:
    if call_surface:
        st.plotly_chart(call_surface, use_container_width=True)
    else:
        st.info("Could not generate call option volatility surface.")
with col2:
    if put_surface:
        st.plotly_chart(put_surface, use_container_width=True)
    else:
        st.info("Could not generate put option volatility surface.")

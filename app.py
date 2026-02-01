import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Var/Vol Swap Pricer & Simulator", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'option_chain' not in st.session_state:
    st.session_state.option_chain = None
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None

# --- HELPER FUNCTIONS ---

def fetch_market_data(ticker, start_date, end_date):
    """Fetches historical underlying data and handles MultiIndex issues."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # FIX: Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if not df.empty and 'Adj Close' in df.columns:
            df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            return df.dropna()
        elif not df.empty and 'Close' in df.columns:
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            return df.dropna()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None

# --- THIS WAS THE MISSING FUNCTION ---
def fetch_option_chain(ticker, expiry_date):
    """Fetches the full option chain (calls and puts) for a specific expiration."""
    tk = yf.Ticker(ticker)
    try:
        # Get specific chain
        chain = tk.option_chain(expiry_date)
        calls = chain.calls
        puts = chain.puts
        
        # Get Current Spot Price
        history = tk.history(period="1d")
        if history.empty:
            return None, None, None
        spot_price = history['Close'].iloc[-1]
        
        return calls, puts, spot_price
    except Exception as e:
        st.error(f"Error fetching options: {e}")
        return None, None, None
# -------------------------------------

def calculate_vix_style_variance(calls, puts, spot_price, days_to_expiry, risk_free_rate=0.045):
    """
    Approximates the Variance Swap Strike (Fair Volatility) using the VIX methodology.
    """
    T = days_to_expiry / 365.0
    if T <= 0: return 0, pd.DataFrame()
    
    # 1. Select OTM Options
    otm_puts = puts[puts['strike'] < spot_price].copy()
    otm_calls = calls[calls['strike'] > spot_price].copy()
    
    # 2. Calculate Mid Price
    otm_puts['bid'] = otm_puts['bid'].fillna(0)
    otm_puts['ask'] = otm_puts['ask'].fillna(0)
    otm_puts['price'] = (otm_puts['bid'] + otm_puts['ask']) / 2
    otm_puts.loc[otm_puts['price'] == 0, 'price'] = otm_puts['lastPrice']
    
    otm_calls['bid'] = otm_calls['bid'].fillna(0)
    otm_calls['ask'] = otm_calls['ask'].fillna(0)
    otm_calls['price'] = (otm_calls['bid'] + otm_calls['ask']) / 2
    otm_calls.loc[otm_calls['price'] == 0, 'price'] = otm_calls['lastPrice']
    
    # 3. Merge and Sort
    df_opts = pd.concat([otm_puts[['strike', 'price']], otm_calls[['strike', 'price']]])
    df_opts = df_opts.sort_values('strike')
    
    # 4. Calculate Contributions
    df_opts['delta_k'] = df_opts['strike'].diff().shift(-1).fillna(0)
    df_opts['contribution'] = (df_opts['delta_k'] / (df_opts['strike']**2)) * np.exp(risk_free_rate * T) * df_opts['price']
    
    sigma_squared = (2 / T) * df_opts['contribution'].sum()
    
    return np.sqrt(sigma_squared) * 100, df_opts

# --- MAIN APP UI ---

st.title("⚡ Variance Swap: Pricing vs. Realized")
st.markdown("Compare the **Implied Volatility** (what the market expects) vs. **Realized Volatility** (what actually happened).")

# SIDEBAR
with st.sidebar:
    st.header("1. Asset Selection")
    ticker = st.text_input("Ticker", value="SPY")
    
    st.header("2. Historical Simulation")
    start_date = st.date_input("History Start", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("History End", value=pd.to_datetime("2023-12-31"))
    
    st.header("3. Live Pricing Params")
    # Default to approx 30 days out
    default_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    expiry_date = st.text_input("Target Expiry (YYYY-MM-DD)", value=default_date)
    st.caption("Note: Must be a valid expiry date for the ticker.")
    
    st.divider()
    fetch_btn = st.button("🚀 ANALYZE / REFRESH DATA", type="primary")

# --- LOGIC CONTROLLER ---

if fetch_btn:
    with st.spinner('Fetching market data and option chains...'):
        # 1. Fetch Historical
        hist_data = fetch_market_data(ticker, start_date, end_date)
        st.session_state.market_data = hist_data
        
        # 2. Fetch Option Chain
        calls, puts, spot = fetch_option_chain(ticker, expiry_date)
        
        if calls is not None and not calls.empty:
            st.session_state.option_chain = {
                'calls': calls,
                'puts': puts,
                'spot': spot,
                'expiry': expiry_date
            }
        else:
            st.warning(f"Could not fetch options for {expiry_date}. Check if date is valid for {ticker} on Yahoo Finance.")
            st.session_state.option_chain = None
            
        st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")

# --- DISPLAY LOGIC ---

if st.session_state.market_data is not None:
    df = st.session_state.market_data
    
    # Realized Vol
    annualization_factor = 252
    df['Squared_Returns'] = df['Log_Return'] ** 2
    df['Cumulative_Var'] = df['Squared_Returns'].cumsum() * (annualization_factor / np.arange(1, len(df) + 1))
    df['Cumulative_Vol'] = np.sqrt(df['Cumulative_Var']) * 100
    realized_vol = df['Cumulative_Vol'].iloc[-1]

    # Implied Vol
    implied_vol_print = "N/A"
    fair_strike_df = pd.DataFrame()
    
    if st.session_state.option_chain:
        data = st.session_state.option_chain
        try:
            d1 = datetime.now()
            d2 = datetime.strptime(data['expiry'], "%Y-%m-%d")
            days_to_expiry = (d2 - d1).days
            
            if days_to_expiry > 0:
                implied_fair_vol, fair_strike_df = calculate_vix_style_variance(
                    data['calls'], data['puts'], data['spot'], days_to_expiry
                )
                implied_vol_print = f"{implied_fair_vol:.2f}%"
            else:
                st.warning("Selected expiration date has passed.")
        except ValueError:
            st.error("Invalid date format. Use YYYY-MM-DD.")

    # Dashboard
    st.write(f"Last Update: {st.session_state.last_fetch_time}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Realized Volatility (Past)", f"{realized_vol:.2f}%")
    col2.metric("Market Implied Volatility (Future)", implied_vol_print)
    
    if implied_vol_print != "N/A":
        diff = realized_vol - float(implied_vol_print.strip('%'))
        col3.metric("Volatility Risk Premium", f"{diff:.2f}%", delta_color="inverse")

    tab1, tab2 = st.tabs(["📉 Historical Path", "🧬 Pricing Replication"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Vol'], mode='lines', name='Realized Vol'))
        if implied_vol_print != "N/A":
             fig.add_hline(y=float(implied_vol_print.strip('%')), line_dash="dash", line_color="red", annotation_text="Implied Price")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if not fair_strike_df.empty:
            st.markdown("### The Replication Strip")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=fair_strike_df['strike'], y=fair_strike_df['contribution'], name='Contribution'))
            st.plotly_chart(fig2, use_container_width=True)
            with st.expander("View Data"):
                st.dataframe(fair_strike_df)
        else:
            st.info("No option data loaded.")
else:
    st.info("👈 Enter parameters and click 'ANALYZE' to start.")

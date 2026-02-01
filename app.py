import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Var/Vol Swap Pricer & Simulator", layout="wide")

# --- SESSION STATE ---
if 'market_data' not in st.session_state: st.session_state.market_data = None
if 'option_chain' not in st.session_state: st.session_state.option_chain = None
if 'last_fetch_time' not in st.session_state: st.session_state.last_fetch_time = None
if 'available_expirations' not in st.session_state: st.session_state.available_expirations = []

# --- HELPER FUNCTIONS ---

def get_expirations(ticker):
    try:
        tk = yf.Ticker(ticker)
        return tk.options
    except:
        return []

def fetch_market_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Handle cases where 'Adj Close' is missing (common in indices)
        col_name = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        
        if not df.empty and col_name in df.columns:
            df['Log_Return'] = np.log(df[col_name] / df[col_name].shift(1))
            return df.dropna()
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def fetch_option_chain(ticker, expiry_date):
    tk = yf.Ticker(ticker)
    try:
        chain = tk.option_chain(expiry_date)
        # Check if chain is actually populated
        if chain.calls.empty or chain.puts.empty:
            return None, None, None
            
        history = tk.history(period="1d")
        spot_price = history['Close'].iloc[-1] if not history.empty else 0
        return chain.calls, chain.puts, spot_price
    except:
        return None, None, None

def calculate_vix_style_variance(calls, puts, spot_price, days_to_expiry, risk_free_rate=0.045):
    T = days_to_expiry / 365.0
    if T <= 0: return 0, pd.DataFrame()
    
    otm_puts = puts[puts['strike'] < spot_price].copy()
    otm_calls = calls[calls['strike'] > spot_price].copy()
    
    # Price cleaning
    for df in [otm_puts, otm_calls]:
        df['bid'] = df['bid'].fillna(0)
        df['ask'] = df['ask'].fillna(0)
        df['price'] = (df['bid'] + df['ask']) / 2
        df.loc[df['price'] == 0, 'price'] = df['lastPrice']
    
    df_opts = pd.concat([otm_puts[['strike', 'price']], otm_calls[['strike', 'price']]])
    df_opts = df_opts.sort_values('strike')
    
    df_opts['delta_k'] = df_opts['strike'].diff().shift(-1).fillna(0)
    df_opts['contribution'] = (df_opts['delta_k'] / (df_opts['strike']**2)) * np.exp(risk_free_rate * T) * df_opts['price']
    
    sigma_squared = (2 / T) * df_opts['contribution'].sum()
    return np.sqrt(sigma_squared) * 100, df_opts

# --- UI ---

st.title("⚡ Variance Swap: Pricing vs. Realized")

with st.sidebar:
    st.header("1. Asset Selection")
    ticker = st.text_input("Ticker", value="^NSEBANK")
    
    # --- EXPIRATION LOGIC ---
    if st.button("Check Expirations"):
        exps = get_expirations(ticker)
        if exps:
            st.session_state.available_expirations = list(exps)
            st.success(f"Found {len(exps)} dates")
        else:
            st.warning("No option chain found in API (Common for NSE Indices)")
    
    expiry_date = None
    if st.session_state.available_expirations:
        expiry_date = st.selectbox("Select Expiry", st.session_state.available_expirations)
    else:
        # If no expirations found, allow manual date for visual purposes
        expiry_date = st.text_input("Target Date (Optional)", value=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))

    st.header("2. Pricing Inputs")
    # NEW: MANUAL OVERRIDE
    manual_iv = st.number_input("Manual Implied Vol (%)", value=0.0, step=0.1, help="If API fails, enter Market IV here (e.g. 14.5)")

    st.header("3. History")
    start_date = st.date_input("Start", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End", value=pd.to_datetime("2023-12-31"))
    
    st.divider()
    fetch_btn = st.button("🚀 ANALYZE", type="primary")

# --- LOGIC ---

if fetch_btn:
    with st.spinner('Fetching data...'):
        st.session_state.market_data = fetch_market_data(ticker, start_date, end_date)
        
        # Try to fetch options
        calls, puts, spot = fetch_option_chain(ticker, expiry_date)
        if calls is not None:
            st.session_state.option_chain = {'calls': calls, 'puts': puts, 'spot': spot, 'expiry': expiry_date}
        else:
            st.session_state.option_chain = None
            if manual_iv == 0:
                st.warning(f"Could not fetch options for {ticker}. Enter 'Manual Implied Vol' in sidebar to proceed.")
        
        st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")

# --- DASHBOARD ---

if st.session_state.market_data is not None:
    df = st.session_state.market_data
    
    # 1. Realized Vol
    df['Squared_Returns'] = df['Log_Return'] ** 2
    df['Cumulative_Var'] = df['Squared_Returns'].cumsum() * (252 / np.arange(1, len(df) + 1))
    df['Cumulative_Vol'] = np.sqrt(df['Cumulative_Var']) * 100
    realized_vol = df['Cumulative_Vol'].iloc[-1]

    # 2. Implied Vol (Auto or Manual)
    implied_vol_val = 0.0
    source = "N/A"
    
    # Try Auto Calculation first
    if st.session_state.option_chain:
        data = st.session_state.option_chain
        try:
            d_days = (datetime.strptime(data['expiry'], "%Y-%m-%d") - datetime.now()).days
            if d_days > 0:
                val, fair_df = calculate_vix_style_variance(data['calls'], data['puts'], data['spot'], d_days)
                implied_vol_val = val
                source = "Market (Calculated)"
        except: pass
    
    # Fallback to Manual
    if implied_vol_val == 0 and manual_iv > 0:
        implied_vol_val = manual_iv
        source = "Manual Input"

    # METRICS
    st.write(f"Last Update: {st.session_state.last_fetch_time} | IV Source: **{source}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Realized Vol", f"{realized_vol:.2f}%")
    c2.metric("Implied Vol", f"{implied_vol_val:.2f}%")
    if implied_vol_val > 0:
        c3.metric("Risk Premium", f"{realized_vol - implied_vol_val:.2f}%", delta_color="inverse")

    tab1, tab2, tab3 = st.tabs(["📉 History", "🧬 Replication", "🔮 Scenario Analysis"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Vol'], name='Realized'))
        if implied_vol_val > 0:
             # FIX: Changed 'color' to 'line_color'
             fig.add_hline(y=implied_vol_val, line_dash="dash", line_color="red", annotation_text="Implied Strike")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.session_state.option_chain:
             st.success("Replication strip loaded from live options.")
             # (Re-run calc just for charting if needed, or store it. Keeping it simple here)
        else:
             st.info("No option chain data available. (Using Manual IV)")

    with tab3:
        if implied_vol_val > 0:
            st.markdown("### P&L Simulator")
            col_a, col_b, col_c = st.columns(3)
            with col_a: vega_notional = st.number_input("Vega Notional ($)", 10000, step=1000)
            with col_b: position = st.radio("Side", ["Long", "Short"], horizontal=True)
            with col_c: sim_vol = st.slider("Simulated Vol", 0.0, 100.0, realized_vol)
            
            K = implied_vol_val
            direction = 1 if position == "Long" else -1
            
            vol_pnl = vega_notional * (sim_vol - K) * direction
            var_notional = vega_notional / (2 * K)
            var_pnl = var_notional * (sim_vol**2 - K**2) * direction
            
            m1, m2 = st.columns(2)
            m1.metric("Vol Swap P&L", f"${vol_pnl:,.0f}")
            m2.metric("Variance Swap P&L", f"${var_pnl:,.0f}", delta=f"${var_pnl-vol_pnl:,.0f} vs Vol Swap")
            
            # Chart
            x = np.linspace(max(0, K-20), K+30, 100)
            y_vol = [vega_notional * (i - K) * direction for i in x]
            y_var = [var_notional * (i**2 - K**2) * direction for i in x]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=x, y=y_vol, name='Vol Swap', line=dict(dash='dash')))
            fig3.add_trace(go.Scatter(x=x, y=y_var, name='Var Swap', line=dict(width=3)))
            
            # FIX: Changed 'color' to 'line_color'
            fig3.add_vline(x=sim_vol, line_color="green", annotation_text="Forecast")
            
            fig3.update_layout(xaxis_title="Vol", yaxis_title="P&L")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Enter Manual IV in sidebar to unlock simulation.")

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
if 'available_expirations' not in st.session_state:
    st.session_state.available_expirations = []

# --- HELPER FUNCTIONS ---

def get_expirations(ticker):
    """Fetches the list of valid expiration dates for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        return tk.options
    except Exception:
        return []

def fetch_market_data(ticker, start_date, end_date):
    """Fetches historical underlying data and handles MultiIndex issues."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
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

def fetch_option_chain(ticker, expiry_date):
    """Fetches the full option chain (calls and puts) for a specific expiration."""
    tk = yf.Ticker(ticker)
    try:
        chain = tk.option_chain(expiry_date)
        calls = chain.calls
        puts = chain.puts
        
        history = tk.history(period="1d")
        if history.empty:
            return None, None, None
        spot_price = history['Close'].iloc[-1]
        
        return calls, puts, spot_price
    except Exception as e:
        st.error(f"Error fetching options: {e}")
        return None, None, None

def calculate_vix_style_variance(calls, puts, spot_price, days_to_expiry, risk_free_rate=0.045):
    """Approximates the Variance Swap Strike (Fair Volatility)."""
    T = days_to_expiry / 365.0
    if T <= 0: return 0, pd.DataFrame()
    
    otm_puts = puts[puts['strike'] < spot_price].copy()
    otm_calls = calls[calls['strike'] > spot_price].copy()
    
    # Calculate Mid Price
    otm_puts['bid'] = otm_puts['bid'].fillna(0)
    otm_puts['ask'] = otm_puts['ask'].fillna(0)
    otm_puts['price'] = (otm_puts['bid'] + otm_puts['ask']) / 2
    otm_puts.loc[otm_puts['price'] == 0, 'price'] = otm_puts['lastPrice']
    
    otm_calls['bid'] = otm_calls['bid'].fillna(0)
    otm_calls['ask'] = otm_calls['ask'].fillna(0)
    otm_calls['price'] = (otm_calls['bid'] + otm_calls['ask']) / 2
    otm_calls.loc[otm_calls['price'] == 0, 'price'] = otm_calls['lastPrice']
    
    # Merge and Sort
    df_opts = pd.concat([otm_puts[['strike', 'price']], otm_calls[['strike', 'price']]])
    df_opts = df_opts.sort_values('strike')
    
    # Calculate Contributions
    df_opts['delta_k'] = df_opts['strike'].diff().shift(-1).fillna(0)
    df_opts['contribution'] = (df_opts['delta_k'] / (df_opts['strike']**2)) * np.exp(risk_free_rate * T) * df_opts['price']
    
    sigma_squared = (2 / T) * df_opts['contribution'].sum()
    
    return np.sqrt(sigma_squared) * 100, df_opts

# --- MAIN APP UI ---

st.title("⚡ Variance Swap: Pricing vs. Realized")

# SIDEBAR
with st.sidebar:
    st.header("1. Asset Selection")
    ticker = st.text_input("Ticker", value="SPY")
    
    if st.button("Check Available Expirations"):
        exps = get_expirations(ticker)
        if exps:
            st.session_state.available_expirations = list(exps)
            st.success(f"Found {len(exps)} expirations!")
        else:
            st.error("No options found. Check ticker.")
    
    if st.session_state.available_expirations:
        expiry_date = st.selectbox("Select Expiry", st.session_state.available_expirations)
    else:
        expiry_date = st.text_input("Or Type Expiry (YYYY-MM-DD)", value=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))

    st.header("2. Historical Simulation")
    start_date = st.date_input("History Start", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("History End", value=pd.to_datetime("2023-12-31"))
    
    st.divider()
    fetch_btn = st.button("🚀 ANALYZE / REFRESH DATA", type="primary")

# --- LOGIC CONTROLLER ---

if fetch_btn:
    with st.spinner('Fetching market data...'):
        hist_data = fetch_market_data(ticker, start_date, end_date)
        st.session_state.market_data = hist_data
        
        calls, puts, spot = fetch_option_chain(ticker, expiry_date)
        
        if calls is not None and not calls.empty:
            st.session_state.option_chain = {
                'calls': calls, 'puts': puts, 'spot': spot, 'expiry': expiry_date
            }
        else:
            st.warning(f"Could not fetch options for {expiry_date}.")
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
    implied_vol_val = 0.0
    implied_vol_print = "N/A"
    fair_strike_df = pd.DataFrame()
    
    if st.session_state.option_chain:
        data = st.session_state.option_chain
        try:
            d1 = datetime.now()
            d2 = datetime.strptime(data['expiry'], "%Y-%m-%d")
            days_to_expiry = (d2 - d1).days
            
            if days_to_expiry > 0:
                implied_vol_val, fair_strike_df = calculate_vix_style_variance(
                    data['calls'], data['puts'], data['spot'], days_to_expiry
                )
                implied_vol_print = f"{implied_vol_val:.2f}%"
            else:
                st.warning("Selected expiration date has passed.")
        except ValueError:
            st.error("Invalid date format.")

    # METRICS ROW
    st.write(f"Last Update: {st.session_state.last_fetch_time}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Realized Volatility (Past)", f"{realized_vol:.2f}%")
    col2.metric("Market Implied Volatility (Future)", implied_vol_print)
    
    if implied_vol_print != "N/A":
        diff = realized_vol - implied_vol_val
        col3.metric("Volatility Risk Premium", f"{diff:.2f}%", delta_color="inverse")

    # --- TABS: HISTORICAL | PRICING | SCENARIO ---
    tab1, tab2, tab3 = st.tabs(["📉 Historical Path", "🧬 Pricing Replication", "🔮 Scenario Analysis"])
    
    # TAB 1: Historical
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Vol'], mode='lines', name='Realized Vol'))
        if implied_vol_val > 0:
             fig.add_hline(y=implied_vol_val, line_dash="dash", line_color="red", annotation_text="Implied Strike")
        st.plotly_chart(fig, use_container_width=True)
        
    # TAB 2: Replication
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

    # TAB 3: P&L Simulator (NEW)
    with tab3:
        st.markdown("### Simulate Your Trade")
        
        if implied_vol_val > 0:
            c1, c2, c3 = st.columns(3)
            with c1:
                vega_notional = st.number_input("Vega Notional ($ per 1% vol)", value=10000, step=1000)
            with c2:
                position = st.radio("Position", ["Long (Buy Vol)", "Short (Sell Vol)"], horizontal=True)
            with c3:
                sim_vol = st.slider("Forecasted Realized Vol", min_value=0.0, max_value=100.0, value=realized_vol)

            # Calculation Logic
            K = implied_vol_val
            direction = 1 if position == "Long (Buy Vol)" else -1
            
            # Vol Swap PnL (Linear)
            # PnL = VegaNotional * (Realized - Strike) * Direction
            vol_pnl = vega_notional * (sim_vol - K) * direction
            
            # Var Swap PnL (Convex)
            # We convert Vega Notional to Variance Notional: N_var = N_vega / (2 * K)
            # Approx Formula: PnL = (N_vega / 2K) * (Realized^2 - Strike^2) * Direction
            var_notional = vega_notional / (2 * K)
            var_pnl = var_notional * (sim_vol**2 - K**2) * direction

            # Display Result
            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Vol Swap P&L", f"${vol_pnl:,.2f}", delta_color="normal")
            m2.metric("Variance Swap P&L", f"${var_pnl:,.2f}", delta=f"${var_pnl - vol_pnl:,.2f} vs Vol Swap")
            
            # P&L Chart
            st.markdown("#### Payout Profile (Convexity Check)")
            
            # Create a range of possible realized volatilities (e.g. 0% to 60%)
            x_range = np.linspace(max(0, K - 20), K + 30, 100)
            
            y_vol = [vega_notional * (x - K) * direction for x in x_range]
            y_var = [var_notional * (x**2 - K**2) * direction for x in x_range]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=x_range, y=y_vol, name='Vol Swap (Linear)', line=dict(dash='dash')))
            fig3.add_trace(go.Scatter(x=x_range, y=y_var, name='Var Swap (Convex)', line=dict(width=3)))
            
            # Mark the current simulation point
            fig3.add_vline(x=sim_vol, line_color="green", annotation_text="Forecast")
            fig3.add_hline(y=0, line_color="gray")
            
            fig3.update_layout(xaxis_title="Realized Volatility (%)", yaxis_title="Profit / Loss ($)")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info(f"💡 **Insight:** Notice how the Variance Swap line curves? That is convexity. If you are Long, you make MORE on the upside and lose LESS on the downside compared to the Vol Swap.")
            
        else:
            st.warning("Please Analyze/Refresh data first to get the Implied Volatility strike price.")

else:
    st.info("👈 Enter parameters and click 'ANALYZE' to start.")

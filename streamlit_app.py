import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date
import datetime

# ── Config ──────────────────────────────────────────────
st.set_page_config(page_title="Terminal | Stock Forecast", layout="wide", initial_sidebar_state="expanded")

# ── CSS Kustom: Bloomberg Terminal Aesthetic ─────────────
st.markdown("""
<style>
    /* Force Deep Black Background for entire app */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #e0e0e0 !important;
    }
    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #333333 !important;
    }
    
    /* Terminal Fonts */
    html, body, p, div, span, h1, h2, h3, h4, h5, h6, table, th, td {
        font-family: 'Courier New', Courier, monospace !important;
    }

    /* Titles & Sections */
    .main-title { color: #ffb900; font-size: 24px; font-weight: bold; border-bottom: 2px solid #333; padding-bottom: 5px; text-transform: uppercase; margin-bottom: 5px; }
    .sub-title { color: #888888; font-size: 13px; margin-bottom: 25px; }
    .section-title { color: #ffb900; font-size: 14px; font-weight: bold; background-color: #111; padding: 4px 8px; border-left: 4px solid #ffb900; margin-top: 2rem; margin-bottom: 1rem; text-transform: uppercase; }
    .sidebar-section { font-size: 12px; font-weight: bold; color: #ffb900; margin-top: 1.5rem; border-bottom: 1px dotted #333; padding-bottom: 2px; margin-bottom: 8px; }
    
    /* Terminal Metric Cards */
    .metric-card { background-color: #050505; border: 1px solid #333333; padding: 12px; text-align: left; }
    .metric-label { font-size: 11px; color: #888888; text-transform: uppercase; }
    .metric-value { font-size: 18px; font-weight: bold; color: #00ff00; margin-top: 4px; }
    .metric-value-cyan { font-size: 18px; font-weight: bold; color: #00ffff; margin-top: 4px; }
    .metric-sub { font-size: 11px; color: #666666; margin-top: 2px; }
    
    /* Terminal Signal Cards */
    .signal-card { background-color: #050505; border: 1px solid #333333; padding: 12px; }
    .badge-buy { color: #00ff00; border: 1px solid #00ff00; padding: 3px 10px; font-weight: bold; font-size: 14px; display: inline-block; background-color: rgba(0,255,0,0.1); }
    .badge-sell { color: #ff0000; border: 1px solid #ff0000; padding: 3px 10px; font-weight: bold; font-size: 14px; display: inline-block; background-color: rgba(255,0,0,0.1); }
    .badge-hold { color: #ffb900; border: 1px solid #ffb900; padding: 3px 10px; font-weight: bold; font-size: 14px; display: inline-block; background-color: rgba(255,185,0,0.1); }
    
    /* Terminal Tables */
    .terminal-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; background-color: #000000; color: #e0e0e0; }
    .terminal-table th { background-color: #111111; color: #ffb900; padding: 8px; text-align: left; border-bottom: 1px solid #ffb900; text-transform: uppercase; }
    .terminal-table td { padding: 8px; border-bottom: 1px dotted #333333; }
    .terminal-table tr:hover { background-color: #0a0a0a; }
    
    /* Buttons */
    div.stButton > button:first-child { background-color: #000000; color: #ffb900; border: 1px solid #ffb900; font-weight: bold; padding: 0.5rem 1rem; text-transform: uppercase; }
    div.stButton > button:first-child:hover { background-color: #ffb900; color: #000000; }
    
    /* Warning Box */
    .terminal-warning { border: 1px solid #ffb900; background-color: rgba(255,185,0,0.05); color: #ffb900; padding: 10px; font-size: 12px; margin-top: 15px; }
    
    /* Hide Streamlit elements that break dark mode */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Data Saham ──────────────────────────────────────────
STOCKS = {
    "Bank BCA (BBCA)": "BBCA.JK",
    "Bank BRI (BBRI)": "BBRI.JK",
    "Bank Mandiri (BMRI)": "BMRI.JK",
}
IPO_DATES = {
    "BBCA.JK": date(2000, 5, 31),
    "BBRI.JK": date(2003, 11, 10),
    "BMRI.JK": date(2003, 7, 14),
}

# ── Sidebar / Controls ───────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#ffb900; margin-bottom: 0; border-bottom: 1px solid #ffb900; padding-bottom: 10px;'>CMD > FORECAST</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'>> INSTRUMENT</div>", unsafe_allow_html=True)
    selected_name = st.selectbox("TICKER", list(STOCKS.keys()), label_visibility="collapsed")
    ticker = STOCKS[selected_name]
    ipo_date = IPO_DATES[ticker]
    
    st.markdown("<div class='sidebar-section'>> DATE RANGE</div>", unsafe_allow_html=True)
    default_start = max(ipo_date, date.today() - datetime.timedelta(days=365*3))
    start_date = st.date_input(
        "START DATE",
        value=default_start,
        min_value=ipo_date,
        max_value=date.today(),
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='sidebar-section'>> HORIZON (DAYS)</div>", unsafe_allow_html=True)
    forecast_days = st.slider(f"Horizon", min_value=7, max_value=90, value=30, label_visibility="collapsed")
    
    st.markdown("<div class='sidebar-section'>> MODEL ENGINE</div>", unsafe_allow_html=True)
    model_choice = st.radio("MODEL", ["Prophet", "SARIMA", "Keduanya"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("EXECUTE RUN", use_container_width=True)

# ── Helper Functions ─────────────────────────────────────
def get_signal_info(forecast_val, last_close, threshold=1.0):
    pct_change = ((forecast_val - last_close) / last_close) * 100
    if pct_change > threshold:
        return "▲ BUY", "badge-buy", pct_change, "#00ff00"
    elif pct_change < -threshold:
        return "▼ SELL", "badge-sell", pct_change, "#ff0000"
    else:
        return "— HOLD", "badge-hold", pct_change, "#ffb900"

def format_signal_html(val):
    if "BUY" in str(val): return f"<span style='color:#00ff00; font-weight:bold;'>{val}</span>"
    if "SELL" in str(val): return f"<span style='color:#ff0000; font-weight:bold;'>{val}</span>"
    if "HOLD" in str(val): return f"<span style='color:#ffb900; font-weight:bold;'>{val}</span>"
    return val

def run_prophet(series_df, periods):
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(series_df)
    future = m.make_future_dataframe(periods=periods, freq="B")
    forecast = m.predict(future)
    return forecast.tail(periods)

def run_sarima(series, periods, last_date):
    sarima = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    sarima_fit = sarima.fit(disp=False)
    sarima_fc = sarima_fit.forecast(steps=periods)
    future_dates = pd.bdate_range(start=last_date, periods=periods + 1)[1:]
    return future_dates, sarima_fc

# ── Main Content ─────────────────────────────────────────
if run:
    with st.spinner("FETCHING DATA & EXECUTING MODEL..."):
        # 1. Ambil Data
        df_raw = yf.download(ticker, start=str(start_date), end=str(date.today()), auto_adjust=True)
        if df_raw.empty:
            st.error("ERR: DATA NOT FOUND.")
            st.stop()

        df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
        df_raw = df_raw.reset_index()
        date_col = df_raw.columns[0]
        df_raw = df_raw.rename(columns={date_col: "Date"})
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])

        last_close = df_raw["Close"].iloc[-1]
        last_open = df_raw["Open"].iloc[-1]
        data_points = len(df_raw)

        # 2. Header Dashboard
        st.markdown(f"<div class='main-title'>EQUITY FORECAST TERMINAL: {ticker}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-title'>SYSTEM: {model_choice.upper()} | HORIZON: {forecast_days} DAYS | STATUS: ONLINE</div>", unsafe_allow_html=True)

        # 3. Baris Metrik Atas 
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (c1, "TICKER", selected_name.split(' (')[0], f"IDX:{ticker.split('.')[0]}", "metric-value-cyan"),
            (c2, "LAST CLOSE", f"Rp {last_close:,.0f}", f"As of {df_raw['Date'].iloc[-1].strftime('%d-%b-%y')}", "metric-value"),
            (c3, "LAST OPEN", f"Rp {last_open:,.0f}", "IDR", "metric-value"),
            (c4, "VOLATILITY DATA", f"{data_points:,}", "Trading Days", "metric-value-cyan"),
            (c5, "FORECAST TARGET", f"+{forecast_days} D", "Forward Looking", "metric-value")
        ]
        
        for col, label, val, sub, val_class in metrics:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='{val_class}'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        # Siapkan data untuk model (Open & Close)
        df_close = df_raw[["Date", "Close"]].dropna().rename(columns={"Date": "ds", "Close": "y"})
        df_open = df_raw[["Date", "Open"]].dropna().rename(columns={"Date": "ds", "Open": "y"})
        df_close["ds"] = pd.to_datetime(df_close["ds"])
        df_open["ds"] = pd.to_datetime(df_open["ds"])

        # Jalankan Model
        fc_prophet_close, fc_prophet_open = None, None
        sarima_fc_close, sarima_fc_open, future_dates_sarima = None, None, None
        
        if model_choice in ["Prophet", "Keduanya"]:
            fc_prophet_close = run_prophet(df_close, forecast_days)
            fc_prophet_open = run_prophet(df_open, forecast_days)
            
        if model_choice in ["SARIMA", "Keduanya"]:
            future_dates_sarima, sarima_fc_close = run_sarima(df_close["y"], forecast_days, df_close["ds"].iloc[-1])
            _, sarima_fc_open = run_sarima(df_open["y"], forecast_days, df_open["ds"].iloc[-1])

        # 4. Trading Signal (H+1)
        st.markdown("<div class='section-title'>SIGNAL ANALYSIS (H+1)</div>", unsafe_allow_html=True)
        sig_c1, sig_c2, sig_c3 = st.columns([1.5, 1.5, 1])
        
        if fc_prophet_close is not None:
            h1_val_p = fc_prophet_close["yhat"].iloc[0]
            sig_text_p, badge_class_p, pct_p, col_p = get_signal_info(h1_val_p, last_close)
            sig_c1.markdown(f"""
            <div class='signal-card'>
                <div class='metric-label' style='margin-bottom:10px;'>ENGINE: PROPHET</div>
                <div class='{badge_class_p}'>{sig_text_p}</div>
                <div style='margin-top:10px; font-size:13px; color:#e0e0e0;'>
                    TARGET T+1 : <span style='color:{col_p}; font-weight:bold;'>Rp {h1_val_p:,.0f}</span><br>
                    VARIANCE   : <span style='color:{col_p};'>{"+" if pct_p>0 else ""}{pct_p:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if sarima_fc_close is not None:
            h1_val_s = sarima_fc_close.iloc[0]
            sig_text_s, badge_class_s, pct_s, col_s = get_signal_info(h1_val_s, last_close)
            sig_c2.markdown(f"""
            <div class='signal-card'>
                <div class='metric-label' style='margin-bottom:10px;'>ENGINE: SARIMA</div>
                <div class='{badge_class_s}'>{sig_text_s}</div>
                <div style='margin-top:10px; font-size:13px; color:#e0e0e0;'>
                    TARGET T+1 : <span style='color:{col_s}; font-weight:bold;'>Rp {h1_val_s:,.0f}</span><br>
                    VARIANCE   : <span style='color:{col_s};'>{"+" if pct_s>0 else ""}{pct_s:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div class='terminal-warning'>SYS_MSG: Signals generated via (T+1 Forecast / Last Close) with ±1.0% threshold. Not financial advice.</div>", unsafe_allow_html=True)

        # 5. Chart
        st.markdown("<div class='section-title'>PRICE CHART TERMINAL</div>", unsafe_allow_html=True)
        fig = go.Figure()
        
        # Plot Histori (Warna Neon Cyan/Abu)
        fig.add_trace(go.Scatter(x=df_close["ds"], y=df_close["y"], name="HIST CLOSE", line=dict(color="#00ffff", width=2)))
        fig.add_trace(go.Scatter(x=df_open["ds"], y=df_open["y"], name="HIST OPEN", line=dict(color="#777777", dash="dot", width=1)))

        # Plot Forecast Prophet (Warna Neon Hijau)
        if fc_prophet_close is not None:
            fig.add_trace(go.Scatter(x=fc_prophet_close["ds"], y=fc_prophet_close["yhat"], name="FCST PROPHET", line=dict(color="#00ff00", dash="dash", width=2)))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_prophet_close["ds"], fc_prophet_close["ds"][::-1]]),
                y=pd.concat([fc_prophet_close["yhat_upper"], fc_prophet_close["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(0,255,0,0.1)", line=dict(color="rgba(255,255,255,0)"),
                name="CONFIDENCE", showlegend=False
            ))
            
        # Plot Forecast SARIMA (Warna Amber/Oranye)
        if sarima_fc_close is not None:
            fig.add_trace(go.Scatter(x=future_dates_sarima, y=sarima_fc_close, name="FCST SARIMA", line=dict(color="#ffb900", dash="dash", width=2)))

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            font=dict(family="Courier New, monospace", color="#e0e0e0", size=11),
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor="#222222", zerolinecolor="#333"),
            yaxis=dict(showgrid=True, gridcolor="#222222", zerolinecolor="#333", tickprefix="Rp ", side="right")
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 6. Tabel Hasil (HTML Murni agar 100% Dark Mode & Bebas Override Streamlit)
        st.markdown("<div class='section-title'>DATA MATRIX (FORWARD LOOKING)</div>", unsafe_allow_html=True)
        
        if fc_prophet_close is not None:
            st.markdown("<div style='color:#00ff00; font-weight:bold; margin-bottom:5px;'>[ PROPHET ENGINE ]</div>", unsafe_allow_html=True)
            tbl = pd.DataFrame({
                "TANGGAL": fc_prophet_close["ds"].dt.strftime("%d-%b-%Y"),
                "FORECAST OPEN": fc_prophet_open["yhat"].values,
                "FORECAST CLOSE": fc_prophet_close["yhat"].values,
                "LOWER BOUND": fc_prophet_close["yhat_lower"].values
            })
            
            signals = []
            prev_val = last_close
            for val in tbl["FORECAST CLOSE"]:
                sig, _, _, _ = get_signal_info(val, prev_val)
                signals.append(sig)
                prev_val = val
                
            tbl["SIGNAL"] = signals
            
            for col in ["FORECAST OPEN", "FORECAST CLOSE", "LOWER BOUND"]:
                tbl[col] = tbl[col].apply(lambda x: f"Rp {x:,.0f}")
                
            tbl["SIGNAL"] = tbl["SIGNAL"].apply(format_signal_html)
            html_table = tbl.to_html(index=False, classes="terminal-table", escape=False)
            st.markdown(html_table, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        if sarima_fc_close is not None:
            st.markdown("<div style='color:#ffb900; font-weight:bold; margin-bottom:5px;'>[ SARIMA ENGINE ]</div>", unsafe_allow_html=True)
            tbl_s = pd.DataFrame({
                "TANGGAL": [d.strftime("%d-%b-%Y") for d in future_dates_sarima],
                "FORECAST OPEN": [f"Rp {v:,.0f}" for v in sarima_fc_open],
                "FORECAST CLOSE": [f"Rp {v:,.0f}" for v in sarima_fc_close]
            })
            
            signals_s = []
            prev_val_s = last_close
            for val in sarima_fc_close:
                sig, _, _, _ = get_signal_info(val, prev_val_s)
                signals_s.append(sig)
                prev_val_s = val
                
            tbl_s["SIGNAL"] = signals_s
            tbl_s["SIGNAL"] = tbl_s["SIGNAL"].apply(format_signal_html)
            
            html_table_s = tbl_s.to_html(index=False, classes="terminal-table", escape=False)
            st.markdown(html_table_s, unsafe_allow_html=True)

        # Footer Terminal
        st.markdown(f"<div style='margin-top:40px; font-size:10px; color:#555555; text-align:right;'>DATA SOURCE: YAHOO FINANCE API | SYS_TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | END OF REPORT</div>", unsafe_allow_html=True)
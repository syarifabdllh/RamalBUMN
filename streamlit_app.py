import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date
import datetime

# ── Config ──────────────────────────────────────────────
st.set_page_config(page_title="Pro Forecast Terminal", layout="wide", initial_sidebar_state="expanded")

# ── CSS Kustom: Modern Dark Financial Dashboard ─────────
st.markdown("""
<style>
    /* Force Deep Dark Background */
    [data-testid="stAppViewContainer"] {
        background-color: #0B0E14 !important;
        color: #D1D5DB !important;
    }
    [data-testid="stHeader"] {
        background-color: #0B0E14 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #11151C !important;
        border-right: 1px solid #1F2937 !important;
    }
    
    /* Modern Clean Font */
    html, body, p, div, span, h1, h2, h3, h4, h5, h6, table, th, td {
        font-family: 'Inter', 'Segoe UI', 'Roboto', Helvetica, Arial, sans-serif !important;
    }

    /* Titles & Sections */
    .main-title { color: #FFFFFF; font-size: 26px; font-weight: 700; margin-bottom: 4px; letter-spacing: -0.5px; }
    .sub-title { color: #9CA3AF; font-size: 13px; margin-bottom: 24px; font-weight: 400; }
    .section-title { color: #F3F4F6; font-size: 13px; font-weight: 600; padding-bottom: 8px; border-bottom: 1px solid #374151; margin-top: 2rem; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 1px; }
    .sidebar-section { font-size: 11px; font-weight: 600; color: #6B7280; margin-top: 1.5rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    
    /* Metric Cards */
    .metric-card { background-color: #151A22; border: 1px solid #1F2937; border-radius: 6px; padding: 16px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .metric-label { font-size: 11px; color: #9CA3AF; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
    .metric-value { font-size: 22px; font-weight: 700; color: #FFFFFF; margin-top: 6px; margin-bottom: 2px; }
    .metric-value-highlight { font-size: 22px; font-weight: 700; color: #3B82F6; margin-top: 6px; margin-bottom: 2px; }
    .metric-sub { font-size: 12px; color: #6B7280; font-weight: 400; }
    
    /* Signal Cards */
    .signal-card { background-color: #151A22; border: 1px solid #1F2937; border-radius: 6px; padding: 16px; }
    .badge-buy { color: #10B981; background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 13px; display: inline-block; }
    .badge-sell { color: #EF4444; background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 13px; display: inline-block; }
    .badge-hold { color: #F59E0B; background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 13px; display: inline-block; }
    
    /* Tables */
    .pro-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; background-color: #0B0E14; color: #D1D5DB; }
    .pro-table th { background-color: #151A22; color: #9CA3AF; padding: 10px 12px; text-align: left; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; border-bottom: 1px solid #374151; border-top: 1px solid #374151; }
    .pro-table td { padding: 10px 12px; border-bottom: 1px solid #1F2937; }
    .pro-table tr:hover { background-color: #11151C; }
    
    /* Buttons */
    div.stButton > button:first-child { background-color: #2563EB; color: #FFFFFF; border: none; border-radius: 6px; font-weight: 600; padding: 0.5rem 1rem; transition: all 0.2s; }
    div.stButton > button:first-child:hover { background-color: #1D4ED8; }
    
    /* Hide specific elements */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Data Saham ──────────────────────────────────────────
STOCKS = {
    "Bank BCA (BBCA)": "BBCA.JK",
    "Bank BRI (BBRI)": "BBRI.JK",
    "Bank Mandiri (BMRI)": "BMRI.JK",
}

# ── Sidebar / Controls ───────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#FFFFFF; margin-bottom: 10px; font-weight: 700; letter-spacing: -0.5px;'>Banking Forecast</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'>INSTRUMENT</div>", unsafe_allow_html=True)
    selected_name = st.selectbox("TICKER", list(STOCKS.keys()), label_visibility="collapsed")
    ticker = STOCKS[selected_name]
    
    st.markdown("<div class='sidebar-section'>FORECAST HORIZON</div>", unsafe_allow_html=True)
    forecast_days = st.slider(f"Horizon", min_value=7, max_value=90, value=30, label_visibility="collapsed")
    
    st.markdown("<div class='sidebar-section'>MODEL ENGINE</div>", unsafe_allow_html=True)
    model_choice = st.radio("MODEL", ["Prophet", "SARIMA", "Keduanya"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Analysis", use_container_width=True)

# ── Helper Functions ─────────────────────────────────────
def get_signal_info(forecast_val, last_close, threshold=1.0):
    pct_change = ((forecast_val - last_close) / last_close) * 100
    if pct_change > threshold:
        return "BUY", "badge-buy", pct_change, "#10B981"
    elif pct_change < -threshold:
        return "SELL", "badge-sell", pct_change, "#EF4444"
    else:
        return "HOLD", "badge-hold", pct_change, "#F59E0B"

def format_signal_html(val):
    if "BUY" in str(val): return f"<span style='color:#10B981; font-weight:600;'>▲ BUY</span>"
    if "SELL" in str(val): return f"<span style='color:#EF4444; font-weight:600;'>▼ SELL</span>"
    if "HOLD" in str(val): return f"<span style='color:#F59E0B; font-weight:600;'>— HOLD</span>"
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
    with st.spinner("Fetching data & running quantitative models..."):
        # 1. Ambil Data (Hardcoded dari awal 2019)
        df_raw = yf.download(ticker, start="2019-01-01", end=str(date.today()), auto_adjust=True)
        if df_raw.empty:
            st.error("Data tidak ditemukan.")
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
        st.markdown(f"<div class='main-title'>{selected_name} Overview</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-title'>Data since 2019-01-01 • Mode: {model_choice} • Projection: {forecast_days} Days</div>", unsafe_allow_html=True)

        # 3. Baris Metrik Atas 
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "TICKER / EXCHANGE", ticker.split('.')[0], "IDX", "metric-value-highlight"),
            (c2, "LAST CLOSE PRICE", f"Rp {last_close:,.0f}", f"As of {df_raw['Date'].iloc[-1].strftime('%d %b %Y')}", "metric-value"),
            (c3, "LAST OPEN PRICE", f"Rp {last_open:,.0f}", "IDR", "metric-value"),
            (c4, "TOTAL TRADING DAYS", f"{data_points:,}", "Since 2019", "metric-value")
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
                <div class='metric-label' style='margin-bottom:12px;'>PROPHET ENGINE</div>
                <div class='{badge_class_p}'>{sig_text_p}</div>
                <div style='margin-top:12px; font-size:13px; color:#D1D5DB;'>
                    Target (T+1): <span style='font-weight:600; color:#FFFFFF;'>Rp {h1_val_p:,.0f}</span> 
                    <span style='color:{col_p}; margin-left:8px;'>{"+" if pct_p>0 else ""}{pct_p:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if sarima_fc_close is not None:
            h1_val_s = sarima_fc_close.iloc[0]
            sig_text_s, badge_class_s, pct_s, col_s = get_signal_info(h1_val_s, last_close)
            sig_c2.markdown(f"""
            <div class='signal-card'>
                <div class='metric-label' style='margin-bottom:12px;'>SARIMA ENGINE</div>
                <div class='{badge_class_s}'>{sig_text_s}</div>
                <div style='margin-top:12px; font-size:13px; color:#D1D5DB;'>
                    Target (T+1): <span style='font-weight:600; color:#FFFFFF;'>Rp {h1_val_s:,.0f}</span> 
                    <span style='color:{col_s}; margin-left:8px;'>{"+" if pct_s>0 else ""}{pct_s:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 5. Chart
        st.markdown("<div class='section-title'>PRICE & PROJECTION CHART</div>", unsafe_allow_html=True)
        fig = go.Figure()
        
        # Plot Histori 
        fig.add_trace(go.Scatter(x=df_close["ds"], y=df_close["y"], name="Historical Close", line=dict(color="#3B82F6", width=2)))
        fig.add_trace(go.Scatter(x=df_open["ds"], y=df_open["y"], name="Historical Open", line=dict(color="#6B7280", dash="dot", width=1.5)))

        # Plot Forecast Prophet 
        if fc_prophet_close is not None:
            fig.add_trace(go.Scatter(x=fc_prophet_close["ds"], y=fc_prophet_close["yhat"], name="Forecast Prophet", line=dict(color="#10B981", dash="dash", width=2.5)))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_prophet_close["ds"], fc_prophet_close["ds"][::-1]]),
                y=pd.concat([fc_prophet_close["yhat_upper"], fc_prophet_close["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(16, 185, 129, 0.1)", line=dict(color="rgba(255,255,255,0)"),
                name="Prophet Confidence", showlegend=False
            ))
            
        # Plot Forecast SARIMA
        if sarima_fc_close is not None:
            fig.add_trace(go.Scatter(x=future_dates_sarima, y=sarima_fc_close, name="Forecast SARIMA", line=dict(color="#F59E0B", dash="dash", width=2.5)))

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0B0E14",
            paper_bgcolor="#0B0E14",
            font=dict(family="Inter, sans-serif", color="#9CA3AF", size=12),
            height=450,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor="#1F2937", zerolinecolor="#1F2937"),
            yaxis=dict(showgrid=True, gridcolor="#1F2937", zerolinecolor="#1F2937", tickprefix="Rp ", side="right")
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 6. Tabel Hasil (HTML Murni)
        st.markdown("<div class='section-title'>PROJECTION MATRIX</div>", unsafe_allow_html=True)
        
        if fc_prophet_close is not None:
            st.markdown("<div style='color:#FFFFFF; font-weight:600; font-size:14px; margin-bottom:10px;'>Prophet Projections</div>", unsafe_allow_html=True)
            tbl = pd.DataFrame({
                "Date": fc_prophet_close["ds"].dt.strftime("%Y-%m-%d"),
                "Forecast Open": fc_prophet_open["yhat"].values,
                "Forecast Close": fc_prophet_close["yhat"].values,
                "Lower Bound": fc_prophet_close["yhat_lower"].values
            })
            
            signals = []
            prev_val = last_close
            for val in tbl["Forecast Close"]:
                sig, _, _, _ = get_signal_info(val, prev_val)
                signals.append(sig)
                prev_val = val
                
            tbl["Signal"] = signals
            
            for col in ["Forecast Open", "Forecast Close", "Lower Bound"]:
                tbl[col] = tbl[col].apply(lambda x: f"Rp {x:,.0f}")
                
            tbl["Signal"] = tbl["Signal"].apply(format_signal_html)
            html_table = tbl.to_html(index=False, classes="pro-table", escape=False)
            st.markdown(html_table, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        if sarima_fc_close is not None:
            st.markdown("<div style='color:#FFFFFF; font-weight:600; font-size:14px; margin-bottom:10px;'>SARIMA Projections</div>", unsafe_allow_html=True)
            tbl_s = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in future_dates_sarima],
                "Forecast Open": [f"Rp {v:,.0f}" for v in sarima_fc_open],
                "Forecast Close": [f"Rp {v:,.0f}" for v in sarima_fc_close]
            })
            
            signals_s = []
            prev_val_s = last_close
            for val in sarima_fc_close:
                sig, _, _, _ = get_signal_info(val, prev_val_s)
                signals_s.append(sig)
                prev_val_s = val
                
            tbl_s["Signal"] = signals_s
            tbl_s["Signal"] = tbl_s["Signal"].apply(format_signal_html)
            
            html_table_s = tbl_s.to_html(index=False, classes="pro-table", escape=False)
            st.markdown(html_table_s, unsafe_allow_html=True)
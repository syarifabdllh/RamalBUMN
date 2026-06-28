import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date
import warnings

warnings.filterwarnings("ignore")

# ── Config & Custom CSS ──────────────────────────────────────────────
st.set_page_config(page_title="Stock Forecast Dark", layout="wide", initial_sidebar_state="expanded")

# CSS Kustom (DARK MODE)
st.markdown("""
    <style>
    /* Global Reset & Background (Dark) */
    .stApp { background-color: #0B0F19; font-family: 'Inter', sans-serif; color: #E2E8F0; }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    
    /* Typography & Headers */
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #9CA3AF;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 16px;
        flex: 1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-title { font-size: 11px; color: #6B7280; font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }
    .kpi-value { font-size: 20px; font-weight: 700; color: #F9FAFB; }
    .kpi-sub { font-size: 12px; color: #9CA3AF; margin-top: 4px; }
    
    /* Signal Cards */
    .signal-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 16px;
        width: 32%;
        display: inline-block;
        margin-right: 15px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 14px;
        margin-top: 5px;
        margin-bottom: 10px;
    }
    /* Dark Mode Badges */
    .badge-buy { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-sell { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-hold { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    
    /* Disclaimer */
    .disclaimer {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #FCD34D;
        margin-bottom: 25px;
    }
    
    /* HTML Table Styling (Dark) */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #111827;
        font-size: 13px;
        color: #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }
    .custom-table th {
        background-color: #1F2937;
        color: #9CA3AF;
        font-weight: 600;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 1px solid #374151;
    }
    .custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #1F2937;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB;
        color: white !important;
        border-radius: 6px;
        padding: 20px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover { background-color: #1D4ED8; }
    </style>
""", unsafe_allow_html=True)

# ── Data & Dictionaries ──────────────────────────────────────────────
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

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color: #F8FAFC; border-bottom: 3px solid #3B82F6; padding-bottom: 10px; margin-bottom: 30px;'>Stock Forecast</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-title'>INSTRUMENT</div>", unsafe_allow_html=True)
    selected_name = st.selectbox("Pilih Saham", list(STOCKS.keys()), label_visibility="collapsed")
    ticker = STOCKS[selected_name]
    
    ipo_date = IPO_DATES[ticker]
    st.markdown(f"<div style='background-color: #1E3A8A; color: #93C5FD; padding: 8px; border-radius: 6px; font-size: 12px; margin-top: -10px; margin-bottom: 20px;'>IPO Date: {ipo_date.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-title'>DATE RANGE</div>", unsafe_allow_html=True)
    start_date = st.date_input("Start Date", value=ipo_date, min_value=ipo_date, max_value=date.today(), label_visibility="collapsed")
    
    st.markdown("<div class='section-title'>FORECAST</div>", unsafe_allow_html=True)
    forecast_days = st.slider("Horizon (days)", min_value=7, max_value=90, value=30, label_visibility="collapsed")
    st.markdown(f"<div style='font-size: 13px; color: #9CA3AF; margin-top: -20px; margin-bottom: 20px;'>Horizon: {forecast_days} days</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-title'>MODEL</div>", unsafe_allow_html=True)
    model_choice = st.radio("Model", ["Prophet", "SARIMA", "Keduanya"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Forecast", use_container_width=True)

# ── Helper Functions (FAST FORECASTING) ──────────────────────────────
@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if not df.empty:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def get_signal_badge(current_price, h1_price):
    pct_change = ((h1_price - current_price) / current_price) * 100
    if pct_change >= 1.0:
        return "▲ BUY", "badge-buy", pct_change
    elif pct_change <= -1.0:
        return "▼ SELL", "badge-sell", pct_change
    else:
        return "— HOLD", "badge-hold", pct_change

def run_fast_prophet(series_df, periods):
    df_train = series_df.tail(300).copy()
    m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False) 
    m.fit(df_train)
    future = m.make_future_dataframe(periods=periods, freq="B")
    return m.predict(future).tail(periods)

def run_fast_sarima(series, periods, last_date):
    model = SARIMAX(series.tail(300), order=(1, 0, 1), seasonal_order=(0, 0, 0, 0))
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=periods)
    dates = pd.bdate_range(start=last_date, periods=periods + 1)[1:]
    return dates, fc

# ── Main Dashboard ───────────────────────────────────────────────────
if run:
    df_raw = get_data(ticker, start_date, date.today())
    if df_raw.empty:
        st.error("Gagal mengambil data.")
        st.stop()

    last_close = df_raw["Close"].iloc[-1]
    last_open  = df_raw["Open"].iloc[-1]
    
    df_close = df_raw[["Date", "Close"]].dropna().rename(columns={"Date": "ds", "Close": "y"})
    df_open = df_raw[["Date", "Open"]].dropna().rename(columns={"Date": "ds", "Open": "y"})

    if model_choice in ["Prophet", "Keduanya"]:
        fc_close_prophet = run_fast_prophet(df_close, forecast_days)
        fc_open_prophet = run_fast_prophet(df_open, forecast_days)
        h1_prophet = fc_close_prophet["yhat"].iloc[0]
        badge_text_p, badge_class_p, pct_p = get_signal_badge(last_close, h1_prophet)
        
    if model_choice in ["SARIMA", "Keduanya"]:
        dates_s, fc_close_sarima = run_fast_sarima(df_close["y"], forecast_days, df_close["ds"].iloc[-1])
        _, fc_open_sarima = run_fast_sarima(df_open["y"], forecast_days, df_open["ds"].iloc[-1])
        h1_sarima = fc_close_sarima.iloc[0]
        badge_text_s, badge_class_s, pct_s = get_signal_badge(last_close, h1_sarima)

    # ── 1. Header ──
    st.markdown("<h1 style='color: #F8FAFC; font-size: 28px; margin-bottom: 5px;'>Forecasting Harga Saham Bank Indonesia</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #9CA3AF; font-size: 14px;'>Model: {model_choice} · Horizon: {forecast_days} hari · Saham: {selected_name}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #1F2937; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # ── 2. KPI Cards ──
    # FIX: Menghapus spasi (indentasi) di awal string agar tidak dirender sebagai code block oleh Markdown
    html_kpi = f"""<div class="kpi-container">
<div class="kpi-card">
<div class="kpi-title">SAHAM</div>
<div class="kpi-value">{selected_name.split('(')[0].strip()}</div>
<div class="kpi-sub">({ticker})</div>
</div>
<div class="kpi-card">
<div class="kpi-title">LAST CLOSE PRICE</div>
<div class="kpi-value">Rp {last_close:,.0f}</div>
</div>
<div class="kpi-card">
<div class="kpi-title">LAST OPEN PRICE</div>
<div class="kpi-value">Rp {last_open:,.0f}</div>
</div>
<div class="kpi-card">
<div class="kpi-title">DATA POINTS</div>
<div class="kpi-value">{len(df_raw):,}</div>
<div class="kpi-sub">hari perdagangan</div>
</div>
<div class="kpi-card">
<div class="kpi-title">FORECAST HORIZON</div>
<div class="kpi-value">{forecast_days} hari</div>
<div class="kpi-sub">ke depan</div>
</div>
</div>"""
    st.markdown(html_kpi, unsafe_allow_html=True)

    # ── 3. Trading Signal ──
    st.markdown("<div class='section-title'>TRADING SIGNAL (H+1)</div>", unsafe_allow_html=True)
    
    # FIX: Menggunakan string satu baris atau tanpa indentasi agar tidak jadi error kotak hitam
    signals_html = "<div>"
    if model_choice in ["Prophet", "Keduanya"]:
        signals_html += f"<div class='signal-card'><div class='kpi-title'>PROPHET — SIGNAL</div><div class='badge {badge_class_p}'>{badge_text_p}</div><div class='kpi-sub'>Forecast H+1: Rp {h1_prophet:,.0f} ({pct_p:+.2f}%)</div></div>"
    if model_choice in ["SARIMA", "Keduanya"]:
        signals_html += f"<div class='signal-card'><div class='kpi-title'>SARIMA — SIGNAL</div><div class='badge {badge_class_s}'>{badge_text_s}</div><div class='kpi-sub'>Forecast H+1: Rp {h1_sarima:,.0f} ({pct_s:+.2f}%)</div></div>"
    signals_html += "</div>"
    
    st.markdown(signals_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div class='disclaimer'>⚠️ <b>Disclaimer:</b> Sinyal buy/sell dihasilkan dari perbandingan forecast Close H+1 vs harga Close terakhir (threshold ±1%). Bukan rekomendasi investasi.</div>", unsafe_allow_html=True)

    # ── 4. Price Chart (Plotly Dark Mode) ──
    st.markdown("<div class='section-title'>PRICE CHART & FORECAST</div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    plot_df = df_raw.tail(150)
    
    # Historical Close & Open
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Historical Close", line=dict(color="#3B82F6", width=2)))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Open"], name="Historical Open", line=dict(color="#64748B", width=1.5, dash="dash")))

    if model_choice in ["Prophet", "Keduanya"]:
        fig.add_trace(go.Scatter(x=fc_close_prophet["ds"], y=fc_close_prophet["yhat"], name="Forecast Close (Prophet)", line=dict(color="#10B981", width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_close_prophet["ds"], fc_close_prophet["ds"][::-1]]),
            y=pd.concat([fc_close_prophet["yhat_upper"], fc_close_prophet["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(16, 185, 129, 0.15)", line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name="Confidence"
        ))

    if model_choice == "SARIMA":
        fig.add_trace(go.Scatter(x=dates_s, y=fc_close_sarima, name="Forecast Close (SARIMA)", line=dict(color="#EF4444", width=2.5, dash="dash")))

    # Update Chart Dark Mode
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#9CA3AF")),
        yaxis=dict(gridcolor="#1F2937", tickformat=",.0f"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="#0B0F19", paper_bgcolor="#0B0F19",
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── 5. Forecast Table ──
    if model_choice in ["Prophet", "Keduanya"]:
        st.markdown("<div class='section-title'>FORECAST TABLE — PROPHET</div>", unsafe_allow_html=True)
        
        # FIX: Format string ditulis 1 baris tanpa spasi/enter agar tidak jadi error Code Block Markdown
        table_html = "<table class='custom-table'><tr><th>TANGGAL</th><th>FORECAST OPEN</th><th>FORECAST CLOSE</th><th>BATAS BAWAH</th><th>SIGNAL</th></tr>"
        
        for i in range(min(5, forecast_days)): 
            tgl = fc_close_prophet.iloc[i]["ds"].strftime("%Y-%m-%d")
            f_open = fc_open_prophet.iloc[i]["yhat"]
            f_close = fc_close_prophet.iloc[i]["yhat"]
            b_bawah = fc_close_prophet.iloc[i]["yhat_lower"]
            
            if i == 0:
                prev_price = last_close
            else:
                prev_price = fc_close_prophet.iloc[i-1]["yhat"]
                
            badge_txt, badge_cls, _ = get_signal_badge(prev_price, f_close)
            
            # Baris tabel (1 baris agar tidak bocor ke markdown formatter)
            table_html += f"<tr><td>{tgl}</td><td>Rp {f_open:,.0f}</td><td>Rp {f_close:,.0f}</td><td>Rp {b_bawah:,.0f}</td><td><span class='badge {badge_cls}' style='margin:0; padding:2px 8px; font-size:12px;'>{badge_txt}</span></td></tr>"
            
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown(f"<p style='text-align: right; color: #4B5563; font-size: 11px; margin-top: 20px;'>Data source: Yahoo Finance · Generated: {date.today().strftime('%d %b %Y')}</p>", unsafe_allow_html=True)

else:
    st.markdown("<h1 style='color: #F8FAFC; font-size: 28px; margin-bottom: 5px;'>Forecasting Harga Saham Bank Indonesia</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #1F2937;'>", unsafe_allow_html=True)
    st.info("Silakan tentukan parameter di sidebar dan klik **Run Forecast**.")
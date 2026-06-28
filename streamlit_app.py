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
st.set_page_config(page_title="Stock Forecast", layout="wide")

# CSS Kustom: Persis seperti referensi Light Mode (Header tidak di-hide agar aman)
st.markdown("""<style>
.stApp { background-color: #F8F9FA; font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.section-title { font-size: 13px; font-weight: 600; color: #64748B; letter-spacing: 1px; margin-top: 25px; margin-bottom: 10px; text-transform: uppercase; }
.kpi-container { display: flex; gap: 15px; margin-bottom: 25px; }
.kpi-card { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; flex: 1; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
.kpi-title { font-size: 11px; color: #94A3B8; font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 20px; font-weight: 700; color: #0F172A; }
.kpi-sub { font-size: 12px; color: #64748B; margin-top: 4px; }
.signal-card { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; width: 32%; display: inline-block; margin-right: 15px; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 14px; margin-top: 5px; margin-bottom: 10px; }
.badge-buy { background-color: #D1FAE5; color: #059669; border: 1px solid #A7F3D0; }
.badge-sell { background-color: #FEE2E2; color: #DC2626; border: 1px solid #FECACA; }
.badge-hold { background-color: #FEF3C7; color: #D97706; border: 1px solid #FDE68A; }
.disclaimer { background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 12px 16px; font-size: 13px; color: #B45309; margin-bottom: 25px; margin-top: 25px; }
.custom-table { width: 100%; border-collapse: collapse; background-color: #FFFFFF; font-size: 13px; margin-bottom: 10px;}
.custom-table th { background-color: #F1F5F9; color: #475569; font-weight: 600; text-align: left; padding: 12px 16px; border-bottom: 1px solid #E2E8F0; }
.custom-table td { padding: 12px 16px; border-bottom: 1px solid #E2E8F0; color: #1E293B; }
.stButton>button { background-color: #10437A; color: white; border-radius: 6px; padding: 15px; font-weight: 600; border: none; margin-top: 10px; }
.stButton>button:hover { background-color: #0c335d; color: white;}
.params-box { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
</style>""", unsafe_allow_html=True)

# ── Data Presets ──────────────────────────────────────────────
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

# ── Helper Functions ──────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if not df.empty:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

def get_signal_badge(current_price, h1_price):
    pct_change = ((h1_price - current_price) / current_price) * 100
    if pct_change >= 1.0: return "▲ BUY", "badge-buy", pct_change
    elif pct_change <= -1.0: return "▼ SELL", "badge-sell", pct_change
    else: return "— HOLD", "badge-hold", pct_change

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

# ── Tampilan Utama (TANPA SIDEBAR) ───────────────────────────────────────────────────
st.markdown("<h1 style='color: #0F172A; font-size: 28px; margin-bottom: 5px;'>Forecasting Harga Saham Bank Indonesia</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 14px;'>Dashboard analisis prediktif pergerakan harga saham menggunakan metode runtun waktu.</p>", unsafe_allow_html=True)

# KOTAK PENGATURAN PARAMETER
st.markdown('<div class="params-box">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_name = st.selectbox("Pilih Saham", list(STOCKS.keys()))
    ticker = STOCKS[selected_name]
with col2:
    start_date = st.date_input("Start Date", value=IPO_DATES[ticker], min_value=IPO_DATES[ticker], max_value=date.today())
with col3:
    forecast_days = st.number_input("Horizon Forecast (Hari)", min_value=7, max_value=90, value=30)
with col4:
    model_choice = st.selectbox("Model", ["Prophet", "SARIMA", "Keduanya"])

run = st.button("🚀 Run Forecast", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── PROSES & HASIL ──
if run:
    with st.spinner("Menghitung forecast..."):
        df_raw = get_data(ticker, start_date, date.today())
        if df_raw is None or df_raw.empty:
            st.error("Gagal mengambil data. Coba lagi.")
            st.stop()

        last_close = df_raw["Close"].iloc[-1]
        last_open  = df_raw["Open"].iloc[-1]
        
        df_close = df_raw[["Date", "Close"]].dropna().rename(columns={"Date": "ds", "Close": "y"})
        df_open = df_raw[["Date", "Open"]].dropna().rename(columns={"Date": "ds", "Open": "y"})

        # Jalankan Model Cepat
        if model_choice in ["Prophet", "Keduanya"]:
            fc_close_prophet = run_fast_prophet(df_close, forecast_days)
            fc_open_prophet = run_fast_prophet(df_open, forecast_days)
            h1_prophet = fc_close_prophet["yhat"].iloc[0]
            badge_text_p, badge_class_p, pct_p = get_signal_badge(last_close, h1_prophet)
            
        if model_choice in ["SARIMA", "Keduanya"]:
            dates_s, fc_close_sarima = run_fast_sarima(df_close["y"], forecast_days, df_close["ds"].iloc[-1])
            h1_sarima = fc_close_sarima.iloc[0]
            badge_text_s, badge_class_s, pct_s = get_signal_badge(last_close, h1_sarima)

        st.markdown("<hr style='border: 1px solid #E2E8F0; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

        # ── 1. KPI Cards ──
        kpi_html = (
            '<div class="kpi-container">'
            '<div class="kpi-card"><div class="kpi-title">SAHAM</div><div class="kpi-value">' + selected_name.split('(')[0].strip() + '</div><div class="kpi-sub">(' + ticker + ')</div></div>'
            '<div class="kpi-card"><div class="kpi-title">LAST CLOSE PRICE</div><div class="kpi-value">Rp ' + f"{last_close:,.0f}" + '</div></div>'
            '<div class="kpi-card"><div class="kpi-title">LAST OPEN PRICE</div><div class="kpi-value">Rp ' + f"{last_open:,.0f}" + '</div></div>'
            '<div class="kpi-card"><div class="kpi-title">DATA POINTS</div><div class="kpi-value">' + f"{len(df_raw):,}" + '</div><div class="kpi-sub">hari perdagangan</div></div>'
            '<div class="kpi-card"><div class="kpi-title">FORECAST HORIZON</div><div class="kpi-value">' + str(forecast_days) + ' hari</div><div class="kpi-sub">ke depan</div></div>'
            '</div>'
        )
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ── 2. Trading Signal ──
        st.markdown("<div class='section-title'>TRADING SIGNAL (H+1)</div>", unsafe_allow_html=True)
        signals_html = '<div>'
        if model_choice in ["Prophet", "Keduanya"]:
            signals_html += '<div class="signal-card"><div class="kpi-title">PROPHET — SIGNAL</div><div class="badge ' + badge_class_p + '">' + badge_text_p + '</div><div class="kpi-sub">Forecast H+1: Rp ' + f"{h1_prophet:,.0f} ({pct_p:+.2f}%)" + '</div></div>'
        if model_choice in ["SARIMA", "Keduanya"]:
            signals_html += '<div class="signal-card"><div class="kpi-title">SARIMA — SIGNAL</div><div class="badge ' + badge_class_s + '">' + badge_text_s + '</div><div class="kpi-sub">Forecast H+1: Rp ' + f"{h1_sarima:,.0f} ({pct_s:+.2f}%)" + '</div></div>'
        signals_html += '</div>'
        st.markdown(signals_html, unsafe_allow_html=True)
        
        st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> Sinyal buy/sell dihasilkan dari perbandingan forecast Close H+1 vs harga Close terakhir (threshold ±1%). Bukan rekomendasi investasi.</div>', unsafe_allow_html=True)

        # ── 3. Price Chart (Plotly) ──
        st.markdown("<div class='section-title'>PRICE CHART & FORECAST</div>", unsafe_allow_html=True)
        
        fig = go.Figure()
        plot_df = df_raw.tail(150)
        
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Historical Close", line=dict(color="#10437A", width=2)))
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Open"], name="Historical Open", line=dict(color="#94A3B8", width=1.5, dash="dash")))

        if model_choice in ["Prophet", "Keduanya"]:
            fig.add_trace(go.Scatter(x=fc_close_prophet["ds"], y=fc_close_prophet["yhat"], name="Forecast Close (Prophet)", line=dict(color="#10B981", width=2.5, dash="dash")))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_close_prophet["ds"], fc_close_prophet["ds"][::-1]]),
                y=pd.concat([fc_close_prophet["yhat_upper"], fc_close_prophet["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(16, 185, 129, 0.1)", line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name="Confidence"
            ))

        if model_choice == "SARIMA":
            fig.add_trace(go.Scatter(x=dates_s, y=fc_close_sarima, name="Forecast Close (SARIMA)", line=dict(color="#EF4444", width=2.5, dash="dash")))

        fig.update_layout(
            template="plotly_white", height=400, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#475569")),
            yaxis=dict(gridcolor="#E2E8F0", tickformat=",.0f"), xaxis=dict(showgrid=False),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # ── 4. Forecast Table ──
        if model_choice in ["Prophet", "Keduanya"]:
            st.markdown("<div class='section-title'>FORECAST TABLE — PROPHET</div>", unsafe_allow_html=True)
            
            table_html = '<table class="custom-table"><tr><th>TANGGAL</th><th>FORECAST OPEN</th><th>FORECAST CLOSE</th><th>BATAS BAWAH</th><th>SIGNAL</th></tr>'
            for i in range(min(5, forecast_days)): 
                tgl = fc_close_prophet.iloc[i]["ds"].strftime("%Y-%m-%d")
                f_open = fc_open_prophet.iloc[i]["yhat"]
                f_close = fc_close_prophet.iloc[i]["yhat"]
                b_bawah = fc_close_prophet.iloc[i]["yhat_lower"]
                
                prev_price = last_close if i == 0 else fc_close_prophet.iloc[i-1]["yhat"]
                badge_txt, badge_cls, _ = get_signal_badge(prev_price, f_close)
                
                table_html += '<tr><td>' + tgl + '</td><td>Rp ' + f"{f_open:,.0f}" + '</td><td>Rp ' + f"{f_close:,.0f}" + '</td><td>Rp ' + f"{b_bawah:,.0f}" + '</td><td><span class="badge ' + badge_cls + '" style="margin:0; padding:2px 8px; font-size:12px;">' + badge_txt + '</span></td></tr>'
                
            table_html += '</table>'
            st.markdown(table_html, unsafe_allow_html=True)

        st.markdown(f"<p style='text-align: right; color: #94A3B8; font-size: 11px; margin-top: 10px;'>Data source: Yahoo Finance &middot; Generated: {date.today().strftime('%d %b %Y')}</p>", unsafe_allow_html=True)
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

# ── Config ──────────────────────────────────────────────
st.set_page_config(page_title="Stock Forecast", layout="wide")

# CSS Kustom (HARDENED LIGHT THEME)
# Menambahkan !important agar kebal dari bocoran Dark Mode bawaan Streamlit
st.markdown("""<style>
/* Memaksa background terang dan teks gelap */
.stApp { background-color: #F8FAFC !important; color: #0F172A !important; font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}

/* Memaksa warna input box dan dropdown agar tetap terang */
.stSelectbox div[data-baseweb="select"] > div, .stDateInput div[data-baseweb="input"] > div, .stNumberInput div[data-baseweb="input"] > div {
    background-color: #FFFFFF !important; color: #0F172A !important; border: 1px solid #CBD5E1 !important;
}

/* Typography & Titles */
.main-title { color: #0F172A !important; font-size: 28px; font-weight: 700; margin-bottom: 0px; }
.sub-title { color: #64748B !important; font-size: 14px; margin-bottom: 20px; }
.section-title { font-size: 13px; font-weight: 700; color: #475569 !important; letter-spacing: 1px; margin-top: 35px; margin-bottom: 15px; text-transform: uppercase; border-bottom: 2px solid #E2E8F0; padding-bottom: 5px;}

/* KPI Cards */
.kpi-container { display: flex; gap: 15px; margin-bottom: 25px; margin-top: 10px; }
.kpi-card { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px; padding: 16px; flex: 1; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.kpi-title { font-size: 11px; color: #64748B !important; font-weight: 700; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 22px; font-weight: 800; color: #0F172A !important; }
.kpi-sub { font-size: 12px; color: #64748B !important; margin-top: 4px; }

/* Signal Cards & Badges */
.signal-card { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px; padding: 16px; width: 32%; display: inline-block; margin-right: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 14px; margin-top: 5px; margin-bottom: 10px; }
.badge-buy { background-color: #D1FAE5 !important; color: #059669 !important; border: 1px solid #A7F3D0 !important; }
.badge-sell { background-color: #FEE2E2 !important; color: #DC2626 !important; border: 1px solid #FECACA !important; }
.badge-hold { background-color: #FEF3C7 !important; color: #D97706 !important; border: 1px solid #FDE68A !important; }

/* Tables */
.custom-table { width: 100%; border-collapse: collapse; background-color: #FFFFFF !important; font-size: 13px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; border: 1px solid #E2E8F0;}
.custom-table th { background-color: #F8FAFC !important; color: #475569 !important; font-weight: 700; text-align: left; padding: 12px 16px; border-bottom: 1px solid #E2E8F0 !important; }
.custom-table td { padding: 12px 16px; border-bottom: 1px solid #E2E8F0 !important; color: #1E293B !important; }

/* Disclaimer Box */
.disclaimer { background-color: #FFFBEB !important; border: 1px solid #FDE68A !important; border-radius: 6px; padding: 12px 16px; font-size: 13px; color: #B45309 !important; margin-bottom: 25px; margin-top: 15px; }
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

# ── Header ─────────────────────────────────────────
st.markdown("<div class='main-title'>Forecasting Harga Saham Bank Indonesia</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Dashboard analisis prediktif pergerakan harga saham menggunakan metode runtun waktu.</div>", unsafe_allow_html=True)

# ── Form Input (Rapi menggunakan st.columns) ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_name = st.selectbox("Instrumen Saham", list(STOCKS.keys()))
    ticker = STOCKS[selected_name]
with col2:
    start_date = st.date_input("Tanggal Mulai", value=IPO_DATES[ticker], min_value=IPO_DATES[ticker], max_value=date.today())
with col3:
    forecast_days = st.number_input("Horizon Forecast (Hari)", min_value=7, max_value=90, value=30)
with col4:
    model_choice = st.selectbox("Algoritma", ["Prophet", "SARIMA", "Keduanya"])

run = st.button("🚀 Jalankan Forecast", type="primary", use_container_width=True)
st.markdown("<hr style='border: 1px solid #E2E8F0; margin-top: 15px;'>", unsafe_allow_html=True)

# ── PROSES & HASIL ──
if run:
    with st.spinner("Mengambil data & menghitung algoritma..."):
        df_raw = get_data(ticker, start_date, date.today())
        if df_raw is None or df_raw.empty:
            st.error("Gagal mengambil data. Coba lagi.")
            st.stop()

        last_close = df_raw["Close"].iloc[-1]
        last_open  = df_raw["Open"].iloc[-1]
        
        df_close = df_raw[["Date", "Close"]].dropna().rename(columns={"Date": "ds", "Close": "y"})
        df_open = df_raw[["Date", "Open"]].dropna().rename(columns={"Date": "ds", "Open": "y"})

        # Jalankan Model
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
        st.markdown("<div class='section-title'>📋 RANGKUMAN TRADING SIGNAL (H+1)</div>", unsafe_allow_html=True)
        signals_html = '<div>'
        if model_choice in ["Prophet", "Keduanya"]:
            signals_html += '<div class="signal-card"><div class="kpi-title">PROPHET — SIGNAL</div><div class="badge ' + badge_class_p + '">' + badge_text_p + '</div><div class="kpi-sub">Forecast H+1: Rp ' + f"{h1_prophet:,.0f} ({pct_p:+.2f}%)" + '</div></div>'
        if model_choice in ["SARIMA", "Keduanya"]:
            signals_html += '<div class="signal-card"><div class="kpi-title">SARIMA — SIGNAL</div><div class="badge ' + badge_class_s + '">' + badge_text_s + '</div><div class="kpi-sub">Forecast H+1: Rp ' + f"{h1_sarima:,.0f} ({pct_s:+.2f}%)" + '</div></div>'
        signals_html += '</div>'
        st.markdown(signals_html, unsafe_allow_html=True)
        st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> Sinyal buy/sell dihasilkan dari perbandingan forecast Close H+1 vs harga Close terakhir (threshold ±1%). Bukan rekomendasi investasi.</div>', unsafe_allow_html=True)

        plot_df = df_raw.tail(150)

        # =====================================================================
        # ── BAGIAN PROPHET ──
        # =====================================================================
        if model_choice in ["Prophet", "Keduanya"]:
            st.markdown("<div class='section-title' style='color:#059669 !important; border-bottom-color:#34D399;'>📈 HASIL FORECAST — MODEL PROPHET</div>", unsafe_allow_html=True)
            
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Historical Close", line=dict(color="#1E3A8A", width=2)))
            fig_p.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Open"], name="Historical Open", line=dict(color="#94A3B8", width=1.5, dash="dash")))
            fig_p.add_trace(go.Scatter(x=fc_close_prophet["ds"], y=fc_close_prophet["yhat"], name="Forecast Close (Prophet)", line=dict(color="#10B981", width=2.5, dash="dash")))
            fig_p.add_trace(go.Scatter(
                x=pd.concat([fc_close_prophet["ds"], fc_close_prophet["ds"][::-1]]),
                y=pd.concat([fc_close_prophet["yhat_upper"], fc_close_prophet["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(16, 185, 129, 0.15)", line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name="Confidence Prophet"
            ))
            
            # THEME=NONE ADALAH KUNCI AGAR CHART TIDAK JADI HITAM KELAM!
            fig_p.update_layout(
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=350, margin=dict(l=0, r=0, t=10, b=0), 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#475569")), 
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#475569")), 
                xaxis=dict(showgrid=False, tickfont=dict(color="#475569"))
            )
            st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False}, theme=None)

            # Tabel
            table_html_p = '<table class="custom-table"><tr><th>TANGGAL</th><th>FORECAST OPEN</th><th>FORECAST CLOSE</th><th>BATAS BAWAH</th><th>SIGNAL</th></tr>'
            for i in range(min(5, forecast_days)): 
                tgl = fc_close_prophet.iloc[i]["ds"].strftime("%Y-%m-%d")
                f_open = fc_open_prophet.iloc[i]["yhat"]
                f_close = fc_close_prophet.iloc[i]["yhat"]
                b_bawah = fc_close_prophet.iloc[i]["yhat_lower"]
                prev_price = last_close if i == 0 else fc_close_prophet.iloc[i-1]["yhat"]
                badge_txt, badge_cls, _ = get_signal_badge(prev_price, f_close)
                table_html_p += '<tr><td>' + tgl + '</td><td>Rp ' + f"{f_open:,.0f}" + '</td><td>Rp ' + f"{f_close:,.0f}" + '</td><td>Rp ' + f"{b_bawah:,.0f}" + '</td><td><span class="badge ' + badge_cls + '" style="margin:0; padding:2px 8px; font-size:12px;">' + badge_txt + '</span></td></tr>'
            table_html_p += '</table>'
            st.markdown(table_html_p, unsafe_allow_html=True)
            if model_choice == "Keduanya": st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================================
        # ── BAGIAN SARIMA ──
        # =====================================================================
        if model_choice in ["SARIMA", "Keduanya"]:
            st.markdown("<div class='section-title' style='color:#DC2626 !important; border-bottom-color:#F87171;'>📈 HASIL FORECAST — MODEL SARIMA</div>", unsafe_allow_html=True)
            
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Historical Close", line=dict(color="#1E3A8A", width=2)))
            fig_s.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Open"], name="Historical Open", line=dict(color="#94A3B8", width=1.5, dash="dash")))
            fig_s.add_trace(go.Scatter(x=dates_s, y=fc_close_sarima, name="Forecast Close (SARIMA)", line=dict(color="#EF4444", width=2.5, dash="dash")))
            
            # MATIKAN THEME OVERRIDE DI SINI JUGA
            fig_s.update_layout(
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=350, margin=dict(l=0, r=0, t=10, b=0), 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#475569")), 
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#475569")), 
                xaxis=dict(showgrid=False, tickfont=dict(color="#475569"))
            )
            st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False}, theme=None)

            table_html_s = '<table class="custom-table"><tr><th>TANGGAL</th><th>FORECAST OPEN</th><th>FORECAST CLOSE</th><th>SIGNAL</th></tr>'
            for i in range(min(5, forecast_days)): 
                tgl = dates_s[i].strftime("%Y-%m-%d")
                f_open = fc_open_sarima.iloc[i]
                f_close = fc_close_sarima.iloc[i]
                prev_price = last_close if i == 0 else fc_close_sarima.iloc[i-1]
                badge_txt, badge_cls, _ = get_signal_badge(prev_price, f_close)
                table_html_s += '<tr><td>' + tgl + '</td><td>Rp ' + f"{f_open:,.0f}" + '</td><td>Rp ' + f"{f_close:,.0f}" + '</td><td><span class="badge ' + badge_cls + '" style="margin:0; padding:2px 8px; font-size:12px;">' + badge_txt + '</span></td></tr>'
            table_html_s += '</table>'
            st.markdown(table_html_s, unsafe_allow_html=True)

        st.markdown(f"<p style='text-align: right; color: #94A3B8 !important; font-size: 11px; margin-top: 30px;'>Data source: Yahoo Finance &middot; Generated: {date.today().strftime('%d %b %Y')}</p>", unsafe_allow_html=True)
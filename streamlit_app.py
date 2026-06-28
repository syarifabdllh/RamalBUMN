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

def run_fast_prophet(series_df, periods):
    df_train = series_df.tail(300).copy() # Batasi 300 hari agar eksekusi instan
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

def generate_signal(current_price, target_price):
    pct = ((target_price - current_price) / current_price) * 100
    if pct >= 1.0:
        return "BUY", pct
    elif pct <= -1.0:
        return "SELL", pct
    else:
        return "HOLD", pct

def style_signal_dataframe(val):
    if val == 'BUY': return 'color: #10B981; font-weight: bold;'
    if val == 'SELL': return 'color: #EF4444; font-weight: bold;'
    return 'color: #F59E0B; font-weight: bold;'

# ── Tampilan Utama (Header) ─────────────────────────────────────────
st.title("📈 Forecasting Harga Saham Bank Indonesia")
st.caption("Dashboard analisis prediktif pergerakan harga saham menggunakan metode runtun waktu (Time-Series).")

# ── Form Input (Menggunakan Native Container) ──
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_name = st.selectbox("Instrumen Saham", list(STOCKS.keys()))
        ticker = STOCKS[selected_name]
    with col2:
        start_date = st.date_input("Tanggal Mulai Data", value=IPO_DATES[ticker], min_value=IPO_DATES[ticker], max_value=date.today())
    with col3:
        forecast_days = st.number_input("Horizon Forecast (Hari)", min_value=7, max_value=90, value=30)
    with col4:
        model_choice = st.selectbox("Pilih Algoritma", ["Prophet", "SARIMA", "Keduanya"])

    run = st.button("🚀 Jalankan Forecast", type="primary", use_container_width=True)

# ── PROSES & HASIL ──
if run:
    with st.spinner("Memproses data & menjalankan algoritma..."):
        df_raw = get_data(ticker, start_date, date.today())
        if df_raw is None or df_raw.empty:
            st.error("Gagal mengambil data. Periksa koneksi atau coba lagi.")
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
            sig_p_text, sig_p_pct = generate_signal(last_close, h1_prophet)
            
        if model_choice in ["SARIMA", "Keduanya"]:
            dates_s, fc_close_sarima = run_fast_sarima(df_close["y"], forecast_days, df_close["ds"].iloc[-1])
            _, fc_open_sarima = run_fast_sarima(df_open["y"], forecast_days, df_open["ds"].iloc[-1])
            h1_sarima = fc_close_sarima.iloc[0]
            sig_s_text, sig_s_pct = generate_signal(last_close, h1_sarima)

        st.divider()

        # ── 1. KPI Cards (Native st.metric) ──
        st.subheader("Informasi Market Terakhir")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Ticker Saham", ticker)
        kpi2.metric("Last Close Price", f"Rp {last_close:,.0f}")
        kpi3.metric("Last Open Price", f"Rp {last_open:,.0f}")
        kpi4.metric("Jumlah Data Historis", f"{len(df_raw):,} Hari")
        
        st.divider()

        # ── 2. Rangkuman Trading Signal ──
        st.subheader("💡 Rangkuman Sinyal Harian (H+1)")
        sig1, sig2, sig3 = st.columns(3)
        
        if model_choice in ["Prophet", "Keduanya"]:
            with sig1:
                st.metric(
                    label="PROPHET SIGNAL", 
                    value=sig_p_text, 
                    delta=f"{sig_p_pct:+.2f}% (Target: Rp {h1_prophet:,.0f})",
                    delta_color="off" if sig_p_text == "HOLD" else "normal"
                )
                
        if model_choice in ["SARIMA", "Keduanya"]:
            with sig2:
                st.metric(
                    label="SARIMA SIGNAL", 
                    value=sig_s_text, 
                    delta=f"{sig_s_pct:+.2f}% (Target: Rp {h1_sarima:,.0f})",
                    delta_color="off" if sig_s_text == "HOLD" else "normal"
                )

        st.info("⚠️ **Disclaimer:** Sinyal BUY/SELL/HOLD ditarik otomatis berdasar proyeksi H+1 vs Harga Penutupan Terakhir (Toleransi ±1%). Bukan rekomendasi investasi absolut.")
        st.divider()

        # Data Potongan untuk visual Chart agar rapi
        plot_df = df_raw.tail(150)

        # =====================================================================
        # ── BAGIAN PROPHET ──
        # =====================================================================
        if model_choice in ["Prophet", "Keduanya"]:
            st.subheader("📉 Hasil Forecast — Model PROPHET")
            
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Historical Close", line=dict(color="#3B82F6", width=2)))
            fig_p.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Open"], name="Historical Open", line=dict(color="#9CA3AF", width=1.5, dash="dash")))
            fig_p.add_trace(go.Scatter(x=fc_close_prophet["ds"], y=fc_close_prophet["yhat"], name="Forecast Prophet", line=dict(color="#10B981", width=2.5, dash="dash")))
            fig_p.add_trace(go.Scatter(
                x=pd.concat([fc_close_prophet["ds"], fc_close_prophet["ds"][::-1]]),
                y=pd.concat([fc_close_prophet["yhat_upper"], fc_close_prophet["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(16, 185, 129, 0.15)", line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name="Confidence"
            ))
            
            # Membiarkan layout menyesuaikan tema browser user secara native
            fig_p.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02), xaxis=dict(showgrid=False))
            st.plotly_chart(fig_p, use_container_width=True, theme="streamlit")

            # Native Pandas Dataframe 
            st.markdown("**Tabel Data Forecast (Prophet):**")
            df_p_table = pd.DataFrame({
                "Tanggal": fc_close_prophet["ds"].dt.strftime("%Y-%m-%d"),
                "Forecast Open": fc_open_prophet["yhat"].values,
                "Forecast Close": fc_close_prophet["yhat"].values,
                "Batas Bawah": fc_close_prophet["yhat_lower"].values
            })
            
            # Hitung signal harian untuk tabel
            sig_list = []
            for i in range(len(df_p_table)):
                prev = last_close if i == 0 else df_p_table["Forecast Close"].iloc[i-1]
                stxt, _ = generate_signal(prev, df_p_table["Forecast Close"].iloc[i])
                sig_list.append(stxt)
            df_p_table["Sinyal"] = sig_list
            
            # Style & Format Dataframe
            styled_p = df_p_table.style.format({
                "Forecast Open": "Rp {:,.0f}",
                "Forecast Close": "Rp {:,.0f}",
                "Batas Bawah": "Rp {:,.0f}"
            }).applymap(style_signal_dataframe, subset=['Sinyal'])
            
            st.dataframe(styled_p, use_container_width=True, hide_index=True)
            
            if model_choice == "Keduanya": st.divider()

        # =====================================================================
        # ── BAGIAN SARIMA ──
        # =====================================================================
        if model_choice in ["SARIMA", "Keduanya"]:
            st.subheader("📉 Hasil Forecast — Model SARIMA")
            
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Historical Close", line=dict(color="#3B82F6", width=2)))
            fig_s.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Open"], name="Historical Open", line=dict(color="#9CA3AF", width=1.5, dash="dash")))
            fig_s.add_trace(go.Scatter(x=dates_s, y=fc_close_sarima, name="Forecast SARIMA", line=dict(color="#EF4444", width=2.5, dash="dash")))
            
            fig_s.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02), xaxis=dict(showgrid=False))
            st.plotly_chart(fig_s, use_container_width=True, theme="streamlit")

            # Native Pandas Dataframe
            st.markdown("**Tabel Data Forecast (SARIMA):**")
            df_s_table = pd.DataFrame({
                "Tanggal": dates_s.strftime("%Y-%m-%d"),
                "Forecast Open": fc_open_sarima.values,
                "Forecast Close": fc_close_sarima.values
            })
            
            # Hitung signal harian untuk tabel
            sig_list_s = []
            for i in range(len(df_s_table)):
                prev = last_close if i == 0 else df_s_table["Forecast Close"].iloc[i-1]
                stxt, _ = generate_signal(prev, df_s_table["Forecast Close"].iloc[i])
                sig_list_s.append(stxt)
            df_s_table["Sinyal"] = sig_list_s
            
            styled_s = df_s_table.style.format({
                "Forecast Open": "Rp {:,.0f}",
                "Forecast Close": "Rp {:,.0f}"
            }).applymap(style_signal_dataframe, subset=['Sinyal'])
            
            st.dataframe(styled_s, use_container_width=True, hide_index=True)
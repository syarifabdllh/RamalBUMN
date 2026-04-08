import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date

# ── Config ──────────────────────────────────────────────
st.set_page_config(page_title="Forecast Saham Bank BUMN", layout="wide")
st.title("📈 Forecasting Harga Saham Bank BUMN Indonesia")

STOCKS = {
    "Bank BRI (BBRI)": "BBRI.JK",
    "Bank BNI (BBNI)": "BBNI.JK",
    "Bank Mandiri (BMRI)": "BMRI.JK",
    "Bank BTN (BBTN)": "BBTN.JK",
    "Bank Syariah Indonesia (BRIS)": "BRIS.JK",
}

# ── Sidebar / Controls ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_name = st.selectbox("Pilih Saham", list(STOCKS.keys()))
    ticker = STOCKS[selected_name]

    start_date = st.date_input("Tanggal Mulai Data", value=date(2022, 1, 1))
    forecast_days = st.slider("Jumlah Hari Forecast", min_value=7, max_value=90, value=30)
    model_choice = st.radio("Model", ["Prophet", "SARIMA", "Keduanya"])
    run = st.button("🚀 Jalankan Forecast", use_container_width=True)

# ── Main ─────────────────────────────────────────────────
if run:
    with st.spinner("Mengambil data & menjalankan model..."):

        # Ambil data
        df_raw = yf.download(ticker, start=str(start_date), end=str(date.today()), auto_adjust=True)
        if df_raw.empty:
            st.error("Data tidak ditemukan. Coba ticker lain.")
            st.stop()

        df = df_raw[["Close"]].dropna().reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])

        # ── Metrics ─────────────────────────────────────
        last_price = df["y"].iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Saham", selected_name)
        col2.metric("Harga Terakhir", f"Rp {last_price:,.0f}")
        col3.metric("Jumlah Data", f"{len(df)} hari")

        st.divider()

        # ── Chart Historical ─────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["y"],
            name="Harga Historis", line=dict(color="#3b82f6")
        ))

        # ── Prophet ─────────────────────────────────────
        if model_choice in ["Prophet", "Keduanya"]:
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=forecast_days, freq="B")
            forecast = m.predict(future)
            forecast_only = forecast.tail(forecast_days)

            fig.add_trace(go.Scatter(
                x=forecast_only["ds"], y=forecast_only["yhat"],
                name="Forecast Prophet", line=dict(color="#10b981", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_only["ds"], forecast_only["ds"][::-1]]),
                y=pd.concat([forecast_only["yhat_upper"], forecast_only["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(16,185,129,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Prophet"
            ))

        # ── SARIMA ──────────────────────────────────────
        if model_choice in ["SARIMA", "Keduanya"]:
            sarima = SARIMAX(df["y"], order=(1,1,1), seasonal_order=(1,1,1,5))
            sarima_fit = sarima.fit(disp=False)
            sarima_forecast = sarima_fit.forecast(steps=forecast_days)
            future_dates = pd.bdate_range(start=df["ds"].iloc[-1], periods=forecast_days+1)[1:]

            fig.add_trace(go.Scatter(
                x=future_dates, y=sarima_forecast,
                name="Forecast SARIMA", line=dict(color="#f59e0b", dash="dash")
            ))

        fig.update_layout(
            title=f"Forecast Harga {selected_name} — {forecast_days} Hari ke Depan",
            xaxis_title="Tanggal", yaxis_title="Harga (IDR)",
            template="plotly_dark", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Tabel Hasil Forecast ─────────────────────────
        if model_choice in ["Prophet", "Keduanya"]:
            st.subheader("📋 Tabel Forecast Prophet")
            tbl = forecast_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            tbl.columns = ["Tanggal", "Prediksi", "Batas Bawah", "Batas Atas"]
            tbl["Tanggal"] = tbl["Tanggal"].dt.strftime("%Y-%m-%d")
            for col in ["Prediksi", "Batas Bawah", "Batas Atas"]:
                tbl[col] = tbl[col].apply(lambda x: f"Rp {x:,.0f}")
            st.dataframe(tbl, use_container_width=True)

        if model_choice in ["SARIMA", "Keduanya"]:
            st.subheader("📋 Tabel Forecast SARIMA")
            tbl_s = pd.DataFrame({
                "Tanggal": [d.strftime("%Y-%m-%d") for d in future_dates],
                "Prediksi": [f"Rp {v:,.0f}" for v in sarima_forecast]
            })
            st.dataframe(tbl_s, use_container_width=True)
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
    "Bank BCA (BBCA)": "BBCA.JK",
    "Bank BRI (BBRI)": "BBRI.JK",
    "Bank Mandiri (BMRI)": "BMRI.JK",
}

# Tanggal IPO masing-masing saham
IPO_DATES = {
    "BBCA.JK": date(2000, 5, 31),   # IPO: 31 Mei 2000, harga IPO Rp 1.400
    "BBRI.JK": date(2003, 11, 10),  # IPO: 10 November 2003, harga IPO Rp 875
    "BMRI.JK": date(2003, 7, 14),   # IPO: 14 Juli 2003, harga IPO Rp 675
}

# ── Sidebar / Controls ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_name = st.selectbox("Pilih Saham", list(STOCKS.keys()))
    ticker = STOCKS[selected_name]

    ipo_date = IPO_DATES[ticker]
    st.info(f"📅 Tanggal IPO: {ipo_date.strftime('%d %B %Y')}")
    start_date = st.date_input(
        "Tanggal Mulai Data",
        value=ipo_date,
        min_value=ipo_date,
        max_value=date.today(),
    )
    forecast_days = st.slider("Jumlah Hari Forecast", min_value=7, max_value=90, value=30)
    price_target = st.radio("Target Harga Forecast", ["Close", "Open", "Keduanya"])
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

        df_raw = df_raw.reset_index()
        df_raw.columns = [c[0] if isinstance(c, tuple) else c for c in df_raw.columns]
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])

        # ── Metrics ─────────────────────────────────────
        last_close = df_raw["Close"].iloc[-1]
        last_open  = df_raw["Open"].iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Saham", selected_name)
        col2.metric("Harga Close Terakhir", f"Rp {last_close:,.0f}")
        col3.metric("Harga Open Terakhir",  f"Rp {last_open:,.0f}")
        col4.metric("Jumlah Data", f"{len(df_raw)} hari")

        st.divider()

        # ── Helper: jalankan Prophet ─────────────────────
        def run_prophet(series_df, periods):
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
            m.fit(series_df)
            future = m.make_future_dataframe(periods=periods, freq="B")
            forecast = m.predict(future)
            return forecast.tail(periods)

        # ── Helper: jalankan SARIMA ──────────────────────
        def run_sarima(series, periods, last_date):
            sarima = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
            sarima_fit = sarima.fit(disp=False)
            sarima_fc = sarima_fit.forecast(steps=periods)
            future_dates = pd.bdate_range(start=last_date, periods=periods + 1)[1:]
            return future_dates, sarima_fc

        # ── Chart ────────────────────────────────────────
        price_modes = []
        if price_target in ["Close", "Keduanya"]:
            price_modes.append("Close")
        if price_target in ["Open", "Keduanya"]:
            price_modes.append("Open")

        color_map = {
            "Close": {"hist": "#3b82f6", "prophet": "#10b981", "sarima": "#f59e0b"},
            "Open":  {"hist": "#a78bfa", "prophet": "#34d399", "sarima": "#fcd34d"},
        }

        fig = go.Figure()
        all_prophet_tbls = {}
        all_sarima_tbls  = {}

        for price_col in price_modes:
            df = df_raw[["Date", price_col]].dropna().copy()
            df.columns = ["ds", "y"]
            df["ds"] = pd.to_datetime(df["ds"])

            # Historical
            fig.add_trace(go.Scatter(
                x=df["ds"], y=df["y"],
                name=f"Historis {price_col}",
                line=dict(color=color_map[price_col]["hist"])
            ))

            # Prophet
            if model_choice in ["Prophet", "Keduanya"]:
                fc = run_prophet(df, forecast_days)
                fig.add_trace(go.Scatter(
                    x=fc["ds"], y=fc["yhat"],
                    name=f"Forecast Prophet ({price_col})",
                    line=dict(color=color_map[price_col]["prophet"], dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                    y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(16,185,129,0.08)" if price_col == "Close" else "rgba(52,211,153,0.08)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"Confidence Prophet ({price_col})",
                    showlegend=True
                ))
                all_prophet_tbls[price_col] = fc

            # SARIMA
            if model_choice in ["SARIMA", "Keduanya"]:
                future_dates, sarima_fc = run_sarima(df["y"], forecast_days, df["ds"].iloc[-1])
                fig.add_trace(go.Scatter(
                    x=future_dates, y=sarima_fc,
                    name=f"Forecast SARIMA ({price_col})",
                    line=dict(color=color_map[price_col]["sarima"], dash="dash")
                ))
                all_sarima_tbls[price_col] = (future_dates, sarima_fc)

        fig.update_layout(
            title=f"Forecast Harga {selected_name} — {forecast_days} Hari ke Depan",
            xaxis_title="Tanggal",
            yaxis_title="Harga (IDR)",
            template="plotly_dark",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Tabel Hasil Forecast ─────────────────────────
        for price_col in price_modes:
            if model_choice in ["Prophet", "Keduanya"] and price_col in all_prophet_tbls:
                st.subheader(f"📋 Tabel Forecast Prophet — {price_col}")
                fc = all_prophet_tbls[price_col]
                tbl = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                tbl.columns = ["Tanggal", "Prediksi", "Batas Bawah", "Batas Atas"]
                tbl["Tanggal"] = tbl["Tanggal"].dt.strftime("%Y-%m-%d")
                for col in ["Prediksi", "Batas Bawah", "Batas Atas"]:
                    tbl[col] = tbl[col].apply(lambda x: f"Rp {x:,.0f}")
                st.dataframe(tbl, use_container_width=True)

            if model_choice in ["SARIMA", "Keduanya"] and price_col in all_sarima_tbls:
                st.subheader(f"📋 Tabel Forecast SARIMA — {price_col}")
                future_dates, sarima_fc = all_sarima_tbls[price_col]
                tbl_s = pd.DataFrame({
                    "Tanggal": [d.strftime("%Y-%m-%d") for d in future_dates],
                    "Prediksi": [f"Rp {v:,.0f}" for v in sarima_fc]
                })
                st.dataframe(tbl_s, use_container_width=True)
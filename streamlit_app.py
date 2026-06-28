import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date
import warnings

# Abaikan warning dari statsmodels agar terminal bersih
warnings.filterwarnings("ignore")

# ── Config & Custom CSS ──────────────────────────────────────────────
st.set_page_config(page_title="Financial Forecasting Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS Kustom untuk tampilan profesional bergaya "Fintech/Trading"
st.markdown("""
    <style>
    /* Menyembunyikan menu bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Styling Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #2B2B40;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Styling Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    
    /* Tombol */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────
st.markdown("<h2 style='text-align: center; color: #E2E8F0; margin-bottom: 30px;'>Quantitative Stock Forecasting Analysis</h2>", unsafe_allow_html=True)

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
    st.markdown("### Parameter Analisis")
    selected_name = st.selectbox("Instrumen Saham", list(STOCKS.keys()))
    ticker = STOCKS[selected_name]

    ipo_date = IPO_DATES[ticker]
    
    start_date = st.date_input(
        "Periode Awal Historis",
        value=date(2020, 1, 1), # Default dibuat lebih baru agar chart tidak terlalu padat
        min_value=ipo_date,
        max_value=date.today(),
    )
    
    forecast_days = st.slider("Horizon Forecast (Hari)", min_value=7, max_value=90, value=30)
    model_choice = st.selectbox("Algoritma Prediksi", ["Prophet", "SARIMA"])
    
    st.markdown("---")
    run = st.button("Generate Forecast & Signal", use_container_width=True)


# ── Helper Functions ─────────────────────────────────────
@st.cache_data(ttl=3600) # Cache agar tidak download terus menerus
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True)
    if not df.empty:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def run_prophet(series_df, periods):
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(series_df)
    future = m.make_future_dataframe(periods=periods, freq="B")
    forecast = m.predict(future)
    return forecast.tail(periods)

def run_sarima(series, periods, last_date):
    sarima = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    sarima_fit = sarima.fit(disp=False)
    sarima_fc = sarima_fit.forecast(steps=periods)
    future_dates = pd.bdate_range(start=last_date, periods=periods + 1)[1:]
    return future_dates, sarima_fc

def generate_signal(current_price, forecasted_prices):
    avg_forecast = np.mean(forecasted_prices)
    price_diff = (avg_forecast - current_price) / current_price * 100
    
    if price_diff >= 3:
        return "STRONG BUY", "#10B981", price_diff # Hijau
    elif 0.5 <= price_diff < 3:
        return "BUY", "#34D399", price_diff # Hijau Muda
    elif -0.5 < price_diff < 0.5:
        return "HOLD", "#FBBF24", price_diff # Kuning
    elif -3 < price_diff <= -0.5:
        return "SELL", "#F87171", price_diff # Merah Muda
    else:
        return "STRONG SELL", "#EF4444", price_diff # Merah

# ── Main ─────────────────────────────────────────────────
if run:
    with st.spinner("Memproses algoritma kuantitatif..."):
        df_raw = get_stock_data(ticker, start_date, date.today())
        
        if df_raw.empty:
            st.error("Data tidak ditemukan. Pastikan koneksi internet stabil.")
            st.stop()

        # Ekstrak data historis terakhir
        last_close = df_raw["Close"].iloc[-1]
        last_open  = df_raw["Open"].iloc[-1]
        prev_close = df_raw["Close"].iloc[-2]
        close_change = last_close - prev_close
        
        # Siapkan data untuk model (Kita forecast Close)
        df_close = df_raw[["Date", "Close"]].dropna().copy()
        df_close.columns = ["ds", "y"]
        df_close["ds"] = pd.to_datetime(df_close["ds"])

        # Jalankan Model
        if model_choice == "Prophet":
            fc_close = run_prophet(df_close, forecast_days)
            forecast_dates = fc_close["ds"]
            forecast_values = fc_close["yhat"].values
        else:
            forecast_dates, forecast_values = run_sarima(df_close["y"], forecast_days, df_close["ds"].iloc[-1])
            
        # Generate Signal
        signal_text, signal_color, percent_change = generate_signal(last_close, forecast_values)

        # ── Tampilan Dashboard Atas (Metrics) ────────────
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Instrumen", selected_name.split(" ")[0], ticker)
        with col2:
            st.metric("Last Close Price", f"Rp {last_close:,.0f}", f"{close_change:,.0f} IDR", delta_color="normal")
        with col3:
            st.metric("Last Open Price",  f"Rp {last_open:,.0f}")
        with col4:
            st.markdown(f"""
                <div style="background-color: {signal_color}20; border: 1px solid {signal_color}; padding: 10px; border-radius: 8px; text-align: center;">
                    <p style="margin:0; font-size: 14px; color: #9CA3AF;">System Signal</p>
                    <h3 style="margin:0; color: {signal_color};">{signal_text}</h3>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs System ──────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📈 Technical & Forecast Chart", "🗃️ Forecast Data", "💡 Signal Analysis"])
        
        with tab1:
            # Menggunakan Candlestick untuk Profesional Look
            fig = go.Figure()

            # Candlestick Historis (Memasukkan Open, High, Low, Close)
            fig.add_trace(go.Candlestick(
                x=df_raw["Date"],
                open=df_raw["Open"], high=df_raw["High"],
                low=df_raw["Low"], close=df_raw["Close"],
                name="Historical Data"
            ))

            # Garis Forecast Close
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_values,
                name=f"Forecast ({model_choice})",
                mode='lines',
                line=dict(color="#3B82F6", width=2, dash="dot")
            ))
            
            # Jika Prophet, tambahkan confidence interval
            if model_choice == "Prophet":
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc_close["ds"], fc_close["ds"][::-1]]),
                    y=pd.concat([fc_close["yhat_upper"], fc_close["yhat_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(59, 130, 246, 0.15)", # Warna biru transparan
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Band"
                ))

            # Kustomisasi Layout ala TradingView
            fig.update_layout(
                title=f"Price Action & Forecast Analysis — {selected_name}",
                yaxis_title="Price (IDR)",
                xaxis_title="Date",
                template="plotly_dark",
                height=600,
                margin=dict(l=0, r=0, t=50, b=0),
                xaxis_rangeslider_visible=False, # Matikan range slider bawaan agar lebih bersih
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="#1E1E2E",
                paper_bgcolor="#1E1E2E"
            )
            
            # Hapus gridlines agar bersih
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='#2B2B40')
            
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Proyeksi Harga Penutupan (Close)")
            if model_choice == "Prophet":
                tbl = fc_close[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                tbl.columns = ["Date", "Predicted Close", "Lower Bound", "Upper Bound"]
            else:
                tbl = pd.DataFrame({
                    "Date": forecast_dates,
                    "Predicted Close": forecast_values
                })
            
            tbl["Date"] = pd.to_datetime(tbl["Date"]).dt.strftime("%Y-%m-%d")
            for col in tbl.columns[1:]:
                tbl[col] = tbl[col].apply(lambda x: f"Rp {x:,.2f}")
            
            # Menggunakan dataframe styling agar lebih profesional
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        with tab3:
            st.markdown("### Analisis Sinyal Algoritmik")
            st.write("Sistem menghasilkan sinyal berdasarkan perbandingan harga penutupan terakhir dengan rata-rata proyeksi harga di masa depan sesuai horizon waktu yang dipilih.")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Model Digunakan:** {model_choice}")
                st.write(f"**Horizon Waktu:** {forecast_days} Hari ke depan")
                st.write(f"**Harga Penutupan Terakhir:** Rp {last_close:,.0f}")
                st.write(f"**Rata-rata Harga Proyeksi:** Rp {np.mean(forecast_values):,.0f}")
            
            with col_b:
                st.write(f"**Potensi Pergerakan (Delta):** {percent_change:,.2f}%")
                st.write(f"**Keputusan Final Sistem:**")
                st.markdown(f"<h2 style='color: {signal_color}; margin-top:0;'>{signal_text}</h2>", unsafe_allow_html=True)
                
            st.info("⚠️ **Disclaimer:** Hasil forecasting dan sinyal ini murni dihasilkan oleh perhitungan algoritma matematis (Quantitative) dan bukan merupakan saran investasi mutlak. Pastikan tetap melakukan analisis fundamental.", icon="ℹ️")
else:
    # Tampilan awal sebelum tombol dijalankan
    st.info("Pilih parameter di sidebar sebelah kiri dan klik **Generate Forecast & Signal** untuk memulai analisis.")
    
    # Ilustrasi kosong (opsional) untuk mempermanis UI saat pertama kali buka
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 300px; color: #6B7280; font-size: 20px;">
        Dashboard Kesiapan Analisis Kuantitatif Pasar Saham
    </div>
    """, unsafe_allow_html=True)
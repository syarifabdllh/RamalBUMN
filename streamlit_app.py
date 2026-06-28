import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date
import datetime

# ── Config ──────────────────────────────────────────────
st.set_page_config(page_title="Forecast Saham Bank", layout="wide", initial_sidebar_state="expanded")

# ── CSS Kustom untuk Tampilan Profesional ────────────────
st.markdown("""
<style>
    /* Styling Sidebar */
    .sidebar-section { font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 1px; margin-top: 1.5rem; margin-bottom: 0.5rem; text-transform: uppercase; }
    
    /* Tombol Utama Sidebar */
    div.stButton > button:first-child { background-color: #0f4a8a; color: white; border-radius: 6px; font-weight: 600; border: none; padding: 0.5rem 1rem; }
    div.stButton > button:first-child:hover { background-color: #0c396b; color: white; }
    
    /* Header Utama */
    .main-title { font-size: 28px; font-weight: 700; color: #0f4a8a; margin-bottom: 0px; padding-bottom: 0px; }
    .sub-title { font-size: 14px; color: #64748b; margin-bottom: 1.5rem; }
    .section-title { font-size: 13px; font-weight: 700; color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px; margin-top: 2rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;}
    
    /* Kartu Metrik (Top Row) */
    .metric-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .metric-label { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 4px; }
    .metric-sub { font-size: 12px; color: #64748b; margin-top: 2px; }
    
    /* Kartu Sinyal H+1 */
    .signal-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }
    .badge-buy { background-color: #d1fae5; color: #065f46; padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 13px; display: inline-block; }
    .badge-sell { background-color: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 13px; display: inline-block; }
    .badge-hold { background-color: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 13px; display: inline-block; }
    
    /* Footer */
    .footer-text { font-size: 11px; color: #94a3b8; text-align: right; margin-top: 2rem; }
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
    st.markdown("<h2 style='color:#0f4a8a; margin-bottom: 0;'>Stock Forecast</h2><hr style='margin-top:0;'>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'>INSTRUMENT</div>", unsafe_allow_html=True)
    selected_name = st.selectbox("Pilih Saham", list(STOCKS.keys()), label_visibility="collapsed")
    ticker = STOCKS[selected_name]
    ipo_date = IPO_DATES[ticker]
    st.info(f"IPO Date: {ipo_date.strftime('%d %b %Y')}")
    
    st.markdown("<div class='sidebar-section'>DATE RANGE</div>", unsafe_allow_html=True)
    start_date = st.date_input(
        "Tanggal Mulai",
        value=ipo_date,
        min_value=ipo_date,
        max_value=date.today(),
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='sidebar-section'>FORECAST</div>", unsafe_allow_html=True)
    forecast_days = st.slider(f"Horizon (days): 30", min_value=7, max_value=90, value=30, label_visibility="collapsed")
    
    st.markdown("<div class='sidebar-section'>MODEL</div>", unsafe_allow_html=True)
    model_choice = st.radio("Pilih Model", ["Prophet", "SARIMA", "Keduanya"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Forecast", use_container_width=True)

# ── Helper Functions ─────────────────────────────────────
def get_signal_info(forecast_val, last_close, threshold=1.0):
    pct_change = ((forecast_val - last_close) / last_close) * 100
    if pct_change > threshold:
        return "▲ BUY", "badge-buy", pct_change
    elif pct_change < -threshold:
        return "▼ SELL", "badge-sell", pct_change
    else:
        return "— HOLD", "badge-hold", pct_change

def run_prophet(series_df, periods):
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(series_df)
    future = m.make_future_dataframe(periods=periods, freq="B")
    forecast = m.predict(future)
    return forecast.tail(periods)

def run_sarima(series, periods, last_date):
    # Disederhanakan untuk stabilitas visualisasi
    sarima = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    sarima_fit = sarima.fit(disp=False)
    sarima_fc = sarima_fit.forecast(steps=periods)
    future_dates = pd.bdate_range(start=last_date, periods=periods + 1)[1:]
    return future_dates, sarima_fc

# ── Main Content ─────────────────────────────────────────
if run:
    with st.spinner("Mengambil data & memproses model..."):
        # 1. Ambil Data
        df_raw = yf.download(ticker, start=str(start_date), end=str(date.today()), auto_adjust=True)
        if df_raw.empty:
            st.error("Data tidak ditemukan. Coba ticker lain.")
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
        st.markdown(f"<div class='main-title'>Forecasting Harga Saham Bank Indonesia</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-title'>Model: {model_choice} • Horizon: {forecast_days} hari • Saham: {selected_name}</div>", unsafe_allow_html=True)

        # 3. Baris Metrik Atas (HTML Kustom)
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (c1, "SAHAM", selected_name.split(' (')[0], f"({ticker})"),
            (c2, "LAST CLOSE PRICE", f"Rp {last_close:,.0f}", ""),
            (c3, "LAST OPEN PRICE", f"Rp {last_open:,.0f}", ""),
            (c4, "DATA POINTS", f"{data_points:,}", "hari perdagangan"),
            (c5, "FORECAST HORIZON", f"{forecast_days} hari", "ke depan")
        ]
        
        for col, label, val, sub in metrics:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        # Siapkan data untuk model
        df = df_raw[["Date", "Close"]].dropna().copy()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])

        # Jalankan Model
        fc_prophet, sarima_fc, future_dates_sarima = None, None, None
        if model_choice in ["Prophet", "Keduanya"]:
            fc_prophet = run_prophet(df, forecast_days)
        if model_choice in ["SARIMA", "Keduanya"]:
            future_dates_sarima, sarima_fc = run_sarima(df["y"], forecast_days, df["ds"].iloc[-1])

        # 4. Trading Signal (H+1)
        st.markdown("<div class='section-title'>TRADING SIGNAL (H+1)</div>", unsafe_allow_html=True)
        sig_c1, sig_c2, sig_c3 = st.columns([1, 1, 2])
        
        if fc_prophet is not None:
            h1_val_p = fc_prophet["yhat"].iloc[0]
            sig_text_p, badge_class_p, pct_p = get_signal_info(h1_val_p, last_close)
            sig_c1.markdown(f"""
            <div class='signal-card'>
                <div class='metric-label' style='margin-bottom:8px;'>PROPHET — SIGNAL</div>
                <div class='{badge_class_p}'>{sig_text_p}</div>
                <div class='metric-sub' style='margin-top:8px;'>Forecast H+1: Rp {h1_val_p:,.0f} ({"+" if pct_p>0 else ""}{pct_p:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        if sarima_fc is not None:
            h1_val_s = sarima_fc.iloc[0]
            sig_text_s, badge_class_s, pct_s = get_signal_info(h1_val_s, last_close)
            sig_c2.markdown(f"""
            <div class='signal-card'>
                <div class='metric-label' style='margin-bottom:8px;'>SARIMA — SIGNAL</div>
                <div class='{badge_class_s}'>{sig_text_s}</div>
                <div class='metric-sub' style='margin-top:8px;'>Forecast H+1: Rp {h1_val_s:,.0f} ({"+" if pct_s>0 else ""}{pct_s:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.warning("⚠️ **Disclaimer:** Sinyal buy/sell dihasilkan dari perbandingan forecast Close H+1 vs harga Close terakhir (threshold ±1%). Bukan rekomendasi investasi.")

        # 5. Chart
        st.markdown("<div class='section-title'>PRICE CHART & FORECAST</div>", unsafe_allow_html=True)
        fig = go.Figure()
        
        # Plot Histori (Batasi 6 bulan terakhir agar chart terlihat rapi)
        df_plot = df.tail(150) 
        fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["y"], name="Historical Close", line=dict(color="#1e3a8a", width=2)))

        if fc_prophet is not None:
            fig.add_trace(go.Scatter(x=fc_prophet["ds"], y=fc_prophet["yhat"], name="Forecast Close (Prophet)", line=dict(color="#10b981", dash="dash", width=2.5)))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_prophet["ds"], fc_prophet["ds"][::-1]]),
                y=pd.concat([fc_prophet["yhat_upper"], fc_prophet["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(16,185,129,0.15)", line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Prophet", showlegend=False
            ))

        if sarima_fc is not None:
            fig.add_trace(go.Scatter(x=future_dates_sarima, y=sarima_fc, name="Forecast Close (SARIMA)", line=dict(color="#f59e0b", dash="dash", width=2.5)))

        fig.update_layout(
            template="plotly_white",
            height=450,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickprefix="Rp ")
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 6. Tabel Hasil
        st.markdown("<div class='section-title'>FORECAST TABLE</div>", unsafe_allow_html=True)
        
        if fc_prophet is not None:
            st.markdown("**PROPHET**")
            tbl = fc_prophet[["ds", "yhat", "yhat_lower"]].copy()
            tbl.columns = ["TANGGAL", "FORECAST CLOSE", "BATAS BAWAH"]
            tbl["TANGGAL"] = tbl["TANGGAL"].dt.strftime("%Y-%m-%d")
            
            # Tambahkan sinyal untuk setiap hari berdasarkan hari sebelumnya
            signals = []
            prev_val = last_close
            for val in tbl["FORECAST CLOSE"]:
                sig, _, _ = get_signal_info(val, prev_val)
                signals.append(sig)
                prev_val = val # update untuk iterasi selanjutnya
                
            tbl["SIGNAL"] = signals
            for col in ["FORECAST CLOSE", "BATAS BAWAH"]:
                tbl[col] = tbl[col].apply(lambda x: f"Rp {x:,.0f}")
                
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        if sarima_fc is not None:
            st.markdown("**SARIMA**")
            tbl_s = pd.DataFrame({
                "TANGGAL": [d.strftime("%Y-%m-%d") for d in future_dates_sarima],
                "FORECAST CLOSE": [f"Rp {v:,.0f}" for v in sarima_fc]
            })
            st.dataframe(tbl_s, use_container_width=True, hide_index=True)

        # Footer
        st.markdown(f"<div class='footer-text'>Data source: Yahoo Finance · Generated: {datetime.datetime.now().strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
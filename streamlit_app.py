import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────
st.set_page_config(page_title="Pro Stock Forecaster", layout="wide", page_icon="📈")

# Custom CSS for Dark Professional Theme
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-box { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-title { font-size: 13px; color: #8B949E; text-transform: uppercase; font-weight: 600; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: 700; color: #58A6FF; }
    .metric-sub { font-size: 13px; color: #8B949E; margin-top: 5px; }
    .signal-buy { color: #3FB950; font-weight: bold; background: rgba(46,160,67,0.15); padding: 5px 10px; border-radius: 5px; border: 1px solid #2EA043; }
    .signal-sell { color: #F85149; font-weight: bold; background: rgba(248,81,73,0.15); padding: 5px 10px; border-radius: 5px; border: 1px solid #F85149; }
    .signal-hold { color: #D29922; font-weight: bold; background: rgba(210,153,34,0.15); padding: 5px 10px; border-radius: 5px; border: 1px solid #D29922; }
    </style>
""", unsafe_allow_html=True)

# ── Data Presets ──────────────────────────────────────────────
PRESET_STOCKS = {
    "Bank BCA (BBCA.JK)": "BBCA.JK",
    "Bank BRI (BBRI.JK)": "BBRI.JK",
    "Bank Mandiri (BMRI.JK)": "BMRI.JK",
    "Telkom (TLKM.JK)": "TLKM.JK",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Apple (AAPL)": "AAPL"
}

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    
    # Allow custom ticker for maximum usability
    stock_mode = st.radio("Metode Input Saham", ["Pilih dari Daftar", "Ketik Simbol (Custom)"])
    if stock_mode == "Pilih dari Daftar":
        ticker = PRESET_STOCKS[st.selectbox("Instrumen", list(PRESET_STOCKS.keys()))]
    else:
        ticker = st.text_input("Masukkan Ticker (ex: GOTO.JK, MSFT)", value="BBCA.JK").upper()
    
    st.markdown("---")
    
    start_date = st.date_input("Mulai Data Historis", value=date.today() - timedelta(days=365 * 3)) # Default 3 years
    forecast_days = st.slider("Horizon Forecast (Hari)", min_value=7, max_value=90, value=30)
    model_choice = st.selectbox("Algoritma Prediksi", ["Prophet", "SARIMA"])
    
    st.markdown("---")
    run = st.button("🚀 Jalankan Analisis", use_container_width=True, type="primary")
    
    st.info("💡 **Tips:** Untuk SARIMA, gunakan data 1-3 tahun terakhir agar proses komputasi lebih stabil dan cepat.")

# ── Core Functions (Robust) ──────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(t, start, end):
    try:
        df = yf.download(t, start=str(start), end=str(end), auto_adjust=True, progress=False)
        if df.empty: return None
        # Fix MultiIndex and Timezone issues (Crucial for Prophet & Web apps)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None) 
        return df
    except Exception:
        return None

def get_signal(current, target):
    pct = ((target - current) / current) * 100
    if pct >= 1.5: return "STRONG BUY", "signal-buy", pct
    elif pct <= -1.5: return "STRONG SELL", "signal-sell", pct
    elif 0.5 <= pct < 1.5: return "BUY", "signal-buy", pct
    elif -1.5 < pct <= -0.5: return "SELL", "signal-sell", pct
    else: return "HOLD", "signal-hold", pct

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# ── Main Execution ───────────────────────────────────────────────────
if run:
    with st.spinner(f"Memproses data & melatih model untuk {ticker}..."):
        df = fetch_data(ticker, start_date, date.today())
        
        if df is None or len(df) < 50:
            st.error(f"❌ Data untuk ticker **{ticker}** tidak ditemukan atau terlalu sedikit (minimal 50 hari). Coba ubah tanggal atau periksa kembali simbol ticker.")
            st.stop()

        # Data preparation
        current_price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2]
        delta_price = current_price - prev_price
        
        # Modeling Data
        df_model = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        
        forecast_dates = []
        forecast_vals = []
        lower_bound = []
        upper_bound = []
        
        # ── Run Algorithm ──
        try:
            if model_choice == "Prophet":
                m = Prophet(daily_seasonality=False, yearly_seasonality=True)
                m.fit(df_model)
                future = m.make_future_dataframe(periods=forecast_days, freq="B")
                forecast = m.predict(future)
                
                # Extract future only
                future_forecast = forecast.tail(forecast_days)
                forecast_dates = future_forecast["ds"]
                forecast_vals = future_forecast["yhat"].values
                lower_bound = future_forecast["yhat_lower"].values
                upper_bound = future_forecast["yhat_upper"].values
                
                # For Evaluation
                hist_pred = forecast.head(len(df_model))["yhat"].values
                mae, rmse, mape = evaluate_model(df_model["y"].values, hist_pred)

            elif model_choice == "SARIMA":
                # Cap training data for SARIMA to prevent infinite looping/crashes
                train_data = df_model["y"].tail(500) 
                model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(0,0,0,0))
                fit_model = model.fit(disp=False)
                
                forecast_vals = fit_model.forecast(steps=forecast_days).values
                # BDateRange to skip weekends
                forecast_dates = pd.bdate_range(start=df_model["ds"].iloc[-1], periods=forecast_days + 1)[1:]
                
                # Mock bounds for SARIMA UI consistency
                std_dev = np.std(train_data) * 0.05
                lower_bound = forecast_vals - std_dev
                upper_bound = forecast_vals + std_dev
                
                # Evaluation (In-sample)
                hist_pred = fit_model.predict(start=0, end=len(train_data)-1).values
                mae, rmse, mape = evaluate_model(train_data.values, hist_pred)
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan pada saat komputasi model algoritma. Detail: {e}")
            st.stop()

        # ── Dashboard Rendering ──
        
        # Title
        st.markdown(f"<h2>Analisis Kuantitatif & Forecasting: <span style='color:#58A6FF;'>{ticker}</span></h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: #30363D; margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-box'><div class='metric-title'>Harga Penutupan (Close)</div><div class='metric-value'>Rp {current_price:,.0f}</div><div class='metric-sub' style='color: {'#3FB950' if delta_price > 0 else '#F85149'};'>{'▲' if delta_price > 0 else '▼'} {delta_price:,.0f} IDR (vs Kemarin)</div></div>", unsafe_allow_html=True)
        
        c2.markdown(f"<div class='metric-box'><div class='metric-title'>Volume Perdagangan</div><div class='metric-value'>{(df['Volume'].iloc[-1] / 1000000):,.1f}M</div><div class='metric-sub'>Lembar Saham</div></div>", unsafe_allow_html=True)
        
        # Signal Engine
        target_price_h1 = forecast_vals[0]
        target_price_end = forecast_vals[-1]
        
        sig_text_1, sig_class_1, pct_1 = get_signal(current_price, target_price_h1)
        sig_text_2, sig_class_2, pct_2 = get_signal(current_price, target_price_end)
        
        c3.markdown(f"<div class='metric-box'><div class='metric-title'>Sinyal Harian (H+1)</div><div style='margin-top: 10px;'><span class='{sig_class_1}'>{sig_text_1}</span></div><div class='metric-sub'>Proyeksi: Rp {target_price_h1:,.0f} ({pct_1:+.2f}%)</div></div>", unsafe_allow_html=True)
        
        c4.markdown(f"<div class='metric-box'><div class='metric-title'>Sinyal Horizon ({forecast_days} Hari)</div><div style='margin-top: 10px;'><span class='{sig_class_2}'>{sig_text_2}</span></div><div class='metric-sub'>Proyeksi: Rp {target_price_end:,.0f} ({pct_2:+.2f}%)</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Professional Interactive Chart (Candlestick + Volume + MA) ──
        st.subheader("📊 Price Action & Proyeksi Tren")
        
        # Calculate MA50
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Slice data for better visualization (last 200 days)
        plot_df = df.tail(200)
        
        # Create subplots (Row 1: Price, Row 2: Volume)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # Candlestick
        fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Historis', increasing_line_color='#3FB950', decreasing_line_color='#F85149'), row=1, col=1)
        
        # MA50
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA50'], line=dict(color='#E3B341', width=1.5), name='MA 50 (Trend)'), row=1, col=1)
        
        # Forecast Line
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_vals, line=dict(color='#58A6FF', width=2, dash='dash'), name=f'Forecast ({model_choice})'), row=1, col=1)
        
        # Forecast Confidence Area
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(forecast_dates), pd.Series(forecast_dates)[::-1]]),
            y=pd.concat([pd.Series(upper_bound), pd.Series(lower_bound)[::-1]]),
            fill='toself', fillcolor='rgba(88, 166, 255, 0.15)', line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band', showlegend=False
        ), row=1, col=1)

        # Volume Bar
        colors = ['#F85149' if row['Open'] - row['Close'] >= 0 else '#3FB950' for index, row in plot_df.iterrows()]
        fig.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
            height=600, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor='#30363D', zeroline=False)
        
        st.plotly_chart(fig, use_container_width=True)

        # ── Bottom Section: Data & Evaluation ──
        col_table, col_eval = st.columns([2, 1])
        
        with col_table:
            st.subheader("📋 Tabel Forecast Lengkap")
            
            # Prepare dataframe
            df_table = pd.DataFrame({
                "Tanggal": forecast_dates,
                "Prediksi Close": forecast_vals,
                "Batas Bawah": lower_bound,
                "Batas Atas": upper_bound
            })
            df_table['Tanggal'] = df_table['Tanggal'].dt.strftime("%Y-%m-%d")
            
            # Generate Signals for Table
            table_signals = []
            for i in range(len(df_table)):
                ref_price = current_price if i == 0 else df_table.loc[i-1, "Prediksi Close"]
                s_txt, _, _ = get_signal(ref_price, df_table.loc[i, "Prediksi Close"])
                table_signals.append(s_txt)
            df_table["Sinyal"] = table_signals
            
            # Styling Pandas DataFrame natively (Responsive & Downloadable)
            def style_signal(val):
                if 'STRONG BUY' in str(val) or val == 'BUY': return 'color: #3FB950; font-weight: bold;'
                elif 'STRONG SELL' in str(val) or val == 'SELL': return 'color: #F85149; font-weight: bold;'
                return 'color: #D29922; font-weight: bold;'
            
            format_dict = {
                "Prediksi Close": "Rp {:,.0f}",
                "Batas Bawah": "Rp {:,.0f}",
                "Batas Atas": "Rp {:,.0f}"
            }
            
            styled_df = df_table.style.format(format_dict).applymap(style_signal, subset=['Sinyal'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=250)

        with col_eval:
            st.subheader("🎯 Akurasi Model Historis")
            st.markdown("<div style='font-size: 14px; color: #8B949E; margin-bottom:15px;'>Seberapa akurat algoritma ini saat dites menggunakan data masa lalu? (Semakin kecil = semakin baik).</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style='background-color: #161B22; padding: 15px; border-radius: 8px; border: 1px solid #30363D;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                        <span style='color: #8B949E;'>MAPE (Error Persentase)</span>
                        <span style='color: #FAFAFA; font-weight: bold;'>{mape:.2f}%</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                        <span style='color: #8B949E;'>MAE (Rata-rata Meleset)</span>
                        <span style='color: #FAFAFA; font-weight: bold;'>Rp {mae:,.0f}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='color: #8B949E;'>RMSE (Standar Deviasi)</span>
                        <span style='color: #FAFAFA; font-weight: bold;'>Rp {rmse:,.0f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Contextual Insight based on MAPE
            if mape < 3: grade, color = "Sangat Baik (Akurat)", "#3FB950"
            elif mape < 10: grade, color = "Cukup Baik", "#D29922"
            else: grade, color = "Buruk (Volatilitas Tinggi)", "#F85149"
            
            st.markdown(f"<div style='margin-top: 15px; padding: 10px; border-left: 4px solid {color}; background-color: rgba(255,255,255,0.05); font-size: 13px;'><b>Kesimpulan:</b> Performa model pada saham ini tergolong <span style='color:{color}; font-weight:bold;'>{grade}</span>.</div>", unsafe_allow_html=True)

    with st.expander("ℹ️ Disclaimer & Metodologi"):
        st.markdown("""
        * **Metodologi Algoritma**: Menggunakan algoritma *Time Series Forecasting* (Prophet dari Meta atau SARIMA dari Statsmodels). Model menganalisis pola musiman, tren masa lalu, dan momentum harga.
        * **Evaluasi**: Hasil evaluasi (MAPE/MAE/RMSE) dihitung dengan membandingkan *fitted values* model terhadap data historis asli.
        * **Disclaimer**: Segala bentuk prediksi yang dihasilkan oleh AI dan perhitungan matematis dalam aplikasi ini **Bukanlah Saran Investasi Finansial (Not Financial Advice)**. Pasar saham sangat fluktuatif dan dipengaruhi oleh sentimen makro ekonomi yang tidak bisa ditangkap oleh data masa lalu semata.
        """)

else:
    # Landing Page State
    st.markdown("<div style='text-align: center; padding: 100px 0;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: #FAFAFA;'>Quant-Engine Forecasting System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8B949E; font-size: 18px;'>Pilih parameter di sidebar sebelah kiri dan jalankan analisis mesin algoritma.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
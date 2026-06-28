import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date

# ── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Forecast", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  .stApp { background-color: #f4f6f9 !important; }

  .main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
    background-color: #f4f6f9 !important;
  }

  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
  }
  [data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stSidebar"] .stButton > button {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
    padding: 0.55rem 1rem !important;
    letter-spacing: 0.02em !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background-color: #16304f !important;
  }
  [data-testid="stSidebar"] label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #64748b !important;
  }
  [data-testid="stSidebar"] [data-baseweb="select"] {
    border-radius: 6px !important;
  }
  [data-testid="stSidebar"] input {
    border-radius: 6px !important;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: 3px solid #1e3a5f !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
  }
  [data-testid="metric-container"] label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #64748b !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.72rem !important;
  }

  /* ── Divider ── */
  hr { border-color: #e2e8f0 !important; margin: 1rem 0 !important; }

  /* ── Section headings ── */
  h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #334155 !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.75rem !important;
  }

  /* ── Tabs ── */
  [data-baseweb="tab-list"] {
    background-color: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 0 !important;
  }
  [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 0 !important;
    background-color: transparent !important;
    border: none !important;
    letter-spacing: 0.03em !important;
  }
  [aria-selected="true"][data-baseweb="tab"] {
    color: #1e3a5f !important;
    border-bottom: 2px solid #1e3a5f !important;
    font-weight: 700 !important;
    background-color: #f8fafc !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
  }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #1e3a5f !important; }

  /* ── Info ── */
  .stInfo {
    background-color: #fffbeb !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 0 6px 6px 0 !important;
    font-size: 0.75rem !important;
    color: #92400e !important;
  }
  .stError {
    background-color: #fef2f2 !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: 0 6px 6px 0 !important;
    font-size: 0.75rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STOCKS = {
    "Bank BCA (BBCA)":     "BBCA.JK",
    "Bank BRI (BBRI)":     "BBRI.JK",
    "Bank Mandiri (BMRI)": "BMRI.JK",
}
IPO_DATES = {
    "BBCA.JK": date(2000, 5, 31),
    "BBRI.JK": date(2003, 11, 10),
    "BMRI.JK": date(2003, 7, 14),
}

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.25rem">
  <div style="font-family:'Inter',sans-serif; font-size:1.6rem; font-weight:700;
    color:#1e293b; line-height:1.2; margin-bottom:0.2rem">
    Stock Forecast
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700;
      color:#1e293b; padding: 0.5rem 0 1rem 0; border-bottom:1px solid #e2e8f0; margin-bottom:1rem">
      ⚙ Pengaturan
    </div>
    """, unsafe_allow_html=True)

    selected_name = st.selectbox("Pilih Saham", list(STOCKS.keys()))
    ticker    = STOCKS[selected_name]
    ipo_date  = IPO_DATES[ticker]

    st.markdown(f"""
    <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:6px;
      padding:0.5rem 0.75rem; font-size:0.72rem; color:#1e40af; margin-bottom:0.75rem;
      font-family:'Inter',sans-serif">
      📅 Tanggal IPO: <strong>{ipo_date.strftime('%d %B %Y')}</strong>
    </div>
    """, unsafe_allow_html=True)

    start_date    = st.date_input("Tanggal Mulai Data", value=ipo_date,
                                   min_value=ipo_date, max_value=date.today())
    forecast_days = st.slider("Horizon Forecast (Hari)", 7, 90, 30)
    model_choice  = st.radio("Model", ["Prophet", "SARIMA", "Keduanya"])

    run = st.button("🚀 Run Forecast", use_container_width=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def get_signal(last_close, fc_next_close, fc_next_open, threshold=0.01):
    chg = (fc_next_close - last_close) / last_close
    if chg > threshold:
        return "BUY", chg * 100
    elif chg < -threshold:
        return "SELL", chg * 100
    else:
        return "HOLD", chg * 100

def prophet_fit(ds_y_df, periods):
    m = Prophet(daily_seasonality=False, yearly_seasonality=True,
                weekly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(ds_y_df)
    future = m.make_future_dataframe(periods=periods, freq="B")
    fc     = m.predict(future)
    return fc.tail(periods).reset_index(drop=True)

def sarima_fit(y_series, periods, last_date):
    mdl  = SARIMAX(y_series, order=(1,1,1), seasonal_order=(1,1,1,5))
    fit  = mdl.fit(disp=False)
    fc   = fit.forecast(steps=periods)
    dates = pd.bdate_range(start=last_date, periods=periods+1)[1:]
    return dates, fc

# ── IDLE STATE ────────────────────────────────────────────────────────────────
if not run:
    st.markdown("""
    <div style="border:2px dashed #e2e8f0; border-radius:10px; padding:4rem 2rem;
      text-align:center; color:#94a3b8; font-family:'Inter',sans-serif;
      font-size:0.85rem; margin-top:2rem; background:#ffffff">
      <div style="font-size:2rem; margin-bottom:0.75rem">📈</div>
      <div style="font-weight:600; color:#475569; margin-bottom:0.4rem">Pilih instrumen dan konfigurasi di sidebar</div>
      <div style="font-size:0.78rem">Tekan <strong>Run Forecast</strong> untuk memulai analisis</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── FETCH & PROCESS ───────────────────────────────────────────────────────────
with st.spinner("Mengambil data & menjalankan model..."):
    df_raw = yf.download(ticker, start=str(start_date), end=str(date.today()), auto_adjust=True)
    if df_raw.empty:
        st.error("Data tidak ditemukan. Coba ubah rentang tanggal.")
        st.stop()

    df_raw.columns = [c[0] if isinstance(c, tuple) else c for c in df_raw.columns]
    df_raw = df_raw.reset_index()
    df_raw = df_raw.rename(columns={df_raw.columns[0]: "Date"})
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.dropna(subset=["Close", "Open"]).copy()

    df_raw["MA20"]   = df_raw["Close"].rolling(20).mean()
    df_raw["MA50"]   = df_raw["Close"].rolling(50).mean()
    df_raw["RSI"]    = compute_rsi(df_raw["Close"])
    df_raw["BB_mid"] = df_raw["Close"].rolling(20).mean()
    df_raw["BB_std"] = df_raw["Close"].rolling(20).std()
    df_raw["BB_up"]  = df_raw["BB_mid"] + 2 * df_raw["BB_std"]
    df_raw["BB_lo"]  = df_raw["BB_mid"] - 2 * df_raw["BB_std"]

    df_close = df_raw[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
    df_open  = df_raw[["Date","Open"]].rename(columns={"Date":"ds","Open":"y"})

    prophet_close, prophet_open, sarima_close, sarima_open = None, None, None, None

    if model_choice in ["Prophet", "Keduanya"]:
        prophet_close = prophet_fit(df_close, forecast_days)
        prophet_open  = prophet_fit(df_open,  forecast_days)
    if model_choice in ["SARIMA", "Keduanya"]:
        last_dt = df_raw["Date"].iloc[-1]
        s_dates_c, s_vals_c = sarima_fit(df_close["y"], forecast_days, last_dt)
        s_dates_o, s_vals_o = sarima_fit(df_open["y"],  forecast_days, last_dt)
        sarima_close = {"dates": s_dates_c, "values": s_vals_c}
        sarima_open  = {"dates": s_dates_o, "values": s_vals_o}

    last_close  = float(df_raw["Close"].iloc[-1])
    last_open   = float(df_raw["Open"].iloc[-1])
    prev_close  = float(df_raw["Close"].iloc[-2])
    chg         = last_close - prev_close
    chg_pct     = chg / prev_close * 100
    last_rsi    = float(df_raw["RSI"].iloc[-1])

    signal_prophet, pct_p = ("—", 0.0)
    signal_sarima,  pct_s = ("—", 0.0)
    fc_p_close_next, fc_p_open_next = last_close, last_open
    fc_s_close_next, fc_s_open_next = last_close, last_open

    if prophet_close is not None:
        fc_p_close_next = float(prophet_close["yhat"].iloc[0])
        fc_p_open_next  = float(prophet_open["yhat"].iloc[0])
        signal_prophet, pct_p = get_signal(last_close, fc_p_close_next, fc_p_open_next)

    if sarima_close is not None:
        fc_s_close_next = float(sarima_close["values"].iloc[0])
        fc_s_open_next  = float(sarima_open["values"].iloc[0])
        signal_sarima, pct_s = get_signal(last_close, fc_s_close_next, fc_s_open_next)

# ── SUBTITLE ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-family:'Inter',sans-serif; font-size:0.78rem; color:#64748b; margin-top:-0.75rem; margin-bottom:1.25rem">
  Model: {model_choice} · Horizon: {forecast_days} hari · Saham: {selected_name}
</div>
""", unsafe_allow_html=True)

# ── KPI METRICS ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Saham",              f"{selected_name.split('(')[0].strip()}\n({ticker})")
c2.metric("Last Close Price",   f"Rp {last_close:,.0f}", f"{chg:+,.0f} ({chg_pct:+.2f}%)")
c3.metric("Last Open Price",    f"Rp {last_open:,.0f}")
c4.metric("Data Points",        f"{len(df_raw):,}",      "hari perdagangan")
c5.metric("Forecast Horizon",   f"{forecast_days} hari", "ke depan")

st.divider()

# ── TRADING SIGNAL ────────────────────────────────────────────────────────────
st.markdown("### Trading Signal (H+1)")

def signal_html(model_name, signal, fc_close, last_close, pct):
    color_map = {
        "BUY":  {"bg":"#f0fdf4","border":"#86efac","badge_bg":"#16a34a","badge_txt":"#ffffff","icon":"▲"},
        "SELL": {"bg":"#fef2f2","border":"#fca5a5","badge_bg":"#dc2626","badge_txt":"#ffffff","icon":"▼"},
        "HOLD": {"bg":"#fffbeb","border":"#fde68a","badge_bg":"#d97706","badge_txt":"#ffffff","icon":"—"},
        "—":    {"bg":"#f8fafc","border":"#e2e8f0","badge_bg":"#94a3b8","badge_txt":"#ffffff","icon":"—"},
    }
    c = color_map.get(signal, color_map["—"])
    fc_txt = f"Forecast H+1: Rp {fc_close:,.0f} ({pct:+.2f}%)" if signal != "—" else "Model tidak dipilih"
    return f"""
    <div style="background:{c['bg']}; border:1.5px solid {c['border']}; border-radius:10px;
      padding:1rem 1.25rem; font-family:'Inter',sans-serif;">
      <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.1em; color:#64748b; margin-bottom:0.6rem">{model_name} — Signal</div>
      <div style="display:inline-flex; align-items:center; gap:0.5rem;
        background:{c['badge_bg']}; color:{c['badge_txt']}; border-radius:6px;
        padding:0.3rem 0.9rem; font-size:0.85rem; font-weight:700; margin-bottom:0.5rem">
        {c['icon']} {signal}
      </div>
      <div style="font-size:0.72rem; color:#475569; margin-top:0.3rem">{fc_txt}</div>
    </div>"""

sig_cols = st.columns(2)
if model_choice in ["Prophet", "Keduanya"]:
    with sig_cols[0]:
        st.markdown(signal_html("Prophet", signal_prophet, fc_p_close_next, last_close, pct_p), unsafe_allow_html=True)
if model_choice in ["SARIMA", "Keduanya"]:
    idx = 1 if model_choice == "Keduanya" else 0
    with sig_cols[idx]:
        st.markdown(signal_html("SARIMA", signal_sarima, fc_s_close_next, last_close, pct_s), unsafe_allow_html=True)

st.info("⚠ Disclaimer: Sinyal buy/sell dihasilkan dari perbandingan forecast Close H+1 vs harga Close terakhir (threshold ±1%). Bukan rekomendasi investasi.")

st.divider()

# ── CHART ─────────────────────────────────────────────────────────────────────
st.markdown("### Price Chart & Forecast")

CHART_COLORS = {
    "close_hist":  "#1e3a5f",
    "open_hist":   "#64748b",
    "prophet_fc":  "#16a34a",
    "sarima_fc":   "#dc2626",
    "ci_fill":     "rgba(22,163,74,0.08)",
    "vline":       "#94a3b8",
}

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_raw["Date"], y=df_raw["Close"],
    name="Historical Close", mode="lines",
    line=dict(color=CHART_COLORS["close_hist"], width=2),
    hovertemplate="Close: Rp %{y:,.0f}<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=df_raw["Date"], y=df_raw["Open"],
    name="Historical Open", mode="lines",
    line=dict(color=CHART_COLORS["open_hist"], width=1.2, dash="dot"),
    hovertemplate="Open: Rp %{y:,.0f}<extra></extra>",
))

if prophet_close is not None:
    fig.add_trace(go.Scatter(
        x=prophet_close["ds"], y=prophet_close["yhat"],
        name="Forecast Close (Prophet)", mode="lines",
        line=dict(color=CHART_COLORS["prophet_fc"], width=2, dash="dash"),
        hovertemplate="Prophet Close: Rp %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([prophet_close["ds"], prophet_close["ds"][::-1]]),
        y=pd.concat([prophet_close["yhat_upper"], prophet_close["yhat_lower"][::-1]]),
        fill="toself", fillcolor=CHART_COLORS["ci_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Interval (Prophet)", hoverinfo="skip",
    ))

if sarima_close is not None:
    fig.add_trace(go.Scatter(
        x=sarima_close["dates"], y=sarima_close["values"],
        name="Forecast Close (SARIMA)", mode="lines",
        line=dict(color=CHART_COLORS["sarima_fc"], width=2, dash="dash"),
        hovertemplate="SARIMA Close: Rp %{y:,.0f}<extra></extra>",
    ))

fig.add_vline(
    x=df_raw["Date"].iloc[-1], line_width=1.5, line_dash="dash",
    line_color=CHART_COLORS["vline"],
    annotation_text="Forecast →", annotation_font_size=10,
    annotation_font_color="#94a3b8", annotation_position="top right",
)

fig.update_layout(
    template="plotly_white",
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", size=11, color="#334155"),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", showline=True, linecolor="#e2e8f0",
               tickfont=dict(size=10, color="#94a3b8"), zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", showline=True, linecolor="#e2e8f0",
               tickfont=dict(size=10, color="#94a3b8"), zeroline=False,
               tickprefix="Rp ", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="left", x=0,
                font=dict(size=10, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=60, r=20, t=50, b=40),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#ffffff", font_size=11, font_family="Inter, sans-serif",
                    bordercolor="#e2e8f0"),
    height=460,
)

st.plotly_chart(fig, use_container_width=True)

# ── OPEN PRICE CHART ──────────────────────────────────────────────────────────
if prophet_open is not None or sarima_open is not None:
    with st.expander("📈 Open Price Forecast Chart", expanded=False):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_raw["Date"], y=df_raw["Open"],
            name="Historical Open", line=dict(color=CHART_COLORS["close_hist"], width=2),
            hovertemplate="Open: Rp %{y:,.0f}<extra></extra>"))
        if prophet_open is not None:
            fig2.add_trace(go.Scatter(x=prophet_open["ds"], y=prophet_open["yhat"],
                name="Forecast Open (Prophet)", line=dict(color=CHART_COLORS["prophet_fc"], width=2, dash="dash"),
                hovertemplate="Prophet Open: Rp %{y:,.0f}<extra></extra>"))
        if sarima_open is not None:
            fig2.add_trace(go.Scatter(x=sarima_open["dates"], y=sarima_open["values"],
                name="Forecast Open (SARIMA)", line=dict(color=CHART_COLORS["sarima_fc"], width=2, dash="dash"),
                hovertemplate="SARIMA Open: Rp %{y:,.0f}<extra></extra>"))
        fig2.add_vline(x=df_raw["Date"].iloc[-1], line_width=1.5, line_dash="dash", line_color=CHART_COLORS["vline"])
        fig2.update_layout(
            template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=10, color="#94a3b8"), zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=10, color="#94a3b8"),
                       zeroline=False, tickprefix="Rp ", tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="top", y=1.08, font=dict(size=10, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=60, r=20, t=40, b=40), height=380, hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── FORECAST TABLE ─────────────────────────────────────────────────────────────
def row_signal(fc_close_val, last_close_val):
    chg = (fc_close_val - last_close_val) / last_close_val
    if chg > 0.01:
        return "▲ BUY"
    elif chg < -0.01:
        return "▼ SELL"
    return "— HOLD"

if model_choice in ["Prophet", "Keduanya"] and prophet_close is not None:
    st.markdown("### Forecast Table — Prophet")
    rows_p = []
    for i in range(len(prophet_close)):
        cl = float(prophet_close["yhat"].iloc[i])
        op = float(prophet_open["yhat"].iloc[i])
        lo = float(prophet_close["yhat_lower"].iloc[i])
        hi = float(prophet_close["yhat_upper"].iloc[i])
        rows_p.append({
            "Tanggal":        prophet_close["ds"].iloc[i].strftime("%Y-%m-%d"),
            "Forecast Open":  f"Rp {op:,.0f}",
            "Forecast Close": f"Rp {cl:,.0f}",
            "Batas Bawah":    f"Rp {lo:,.0f}",
            "Batas Atas":     f"Rp {hi:,.0f}",
            "Signal":         row_signal(cl, last_close),
        })
    st.dataframe(pd.DataFrame(rows_p), use_container_width=True, height=320, hide_index=True)

if model_choice in ["SARIMA", "Keduanya"] and sarima_close is not None:
    st.markdown("### Forecast Table — SARIMA")
    rows_s = []
    for i in range(len(sarima_close["dates"])):
        cl = float(sarima_close["values"].iloc[i])
        op = float(sarima_open["values"].iloc[i])
        rows_s.append({
            "Tanggal":        sarima_close["dates"][i].strftime("%Y-%m-%d"),
            "Forecast Open":  f"Rp {op:,.0f}",
            "Forecast Close": f"Rp {cl:,.0f}",
            "Batas Bawah":    "—",
            "Batas Atas":     "—",
            "Signal":         row_signal(cl, last_close),
        })
    st.dataframe(pd.DataFrame(rows_s), use_container_width=True, height=320, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-family:'Inter',sans-serif; font-size:0.65rem; color:#94a3b8;
  border-top:1px solid #e2e8f0; padding-top:0.75rem; margin-top:1.5rem;
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem">
  <span>Data source: Yahoo Finance · Model: Prophet + SARIMA · Untuk keperluan akademis</span>
  <span>Generated: {date.today().strftime("%d %b %Y")}</span>
</div>
""", unsafe_allow_html=True)
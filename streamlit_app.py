import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Bloomberg-style Light Mode ───────────────────────────────────
st.markdown("""
<style>
    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        background-color: #F7F8FA;
        color: #1A1A2E;
    }
    .stApp { background-color: #F7F8FA; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* ---- Sidebar labels ---- */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
    }

    /* ---- Sidebar brand header ---- */
    .sidebar-brand {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F4C81;
        letter-spacing: -0.01em;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #0F4C81;
        margin-bottom: 1.2rem;
    }
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin: 1.2rem 0 0.4rem 0;
    }

    /* ---- Page header ---- */
    .page-header {
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .page-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F4C81;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 0.15rem;
    }

    /* ---- Metric Cards ---- */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        min-height: 90px;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1A1A2E;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* ---- Signal Badge ---- */
    .signal-BUY {
        background: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .signal-SELL {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .signal-HOLD {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #E2E8F0;
        margin: 1.6rem 0 0.8rem 0;
    }

    /* ---- Tables ---- */
    .stDataFrame { border: 1px solid #E2E8F0; border-radius: 8px; overflow: hidden; }
    .stDataFrame thead { background-color: #F1F5F9; }
    .stDataFrame th {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
    }

    /* ---- Run button ---- */
    .stButton > button {
        background-color: #0F4C81;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 0.55rem 1rem;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #1A6DB5; }

    /* ---- Info box ---- */
    .info-box {
        background: #EFF6FF;
        border-left: 3px solid #3B82F6;
        border-radius: 4px;
        padding: 0.6rem 0.9rem;
        font-size: 0.78rem;
        color: #1E40AF;
        margin-bottom: 0.8rem;
    }

    /* ---- Disclaimer ---- */
    .disclaimer {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.75rem;
        color: #78350F;
        margin-top: 1rem;
    }

    /* ---- Divider ---- */
    hr { border: none; border-top: 1px solid #E2E8F0; margin: 1.2rem 0; }

    /* ---- Hide default Streamlit chrome ---- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
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
BUY_THRESHOLD  = 0.01   # forecast naik > 1%  → BUY
SELL_THRESHOLD = -0.01  # forecast turun > 1% → SELL

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Stock Forecast</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Instrument</div>', unsafe_allow_html=True)
    selected_name = st.selectbox("Pilih Saham", list(STOCKS.keys()), label_visibility="collapsed")
    ticker = STOCKS[selected_name]
    ipo_date = IPO_DATES[ticker]

    st.markdown(
        f'<div class="info-box">IPO Date: {ipo_date.strftime("%d %b %Y")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Date Range</div>', unsafe_allow_html=True)
    start_date = st.date_input(
        "Start Date",
        value=ipo_date,
        min_value=ipo_date,
        max_value=date.today(),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">Forecast</div>', unsafe_allow_html=True)
    forecast_days = st.slider("Horizon (days)", min_value=7, max_value=90, value=30)

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Model",
        ["Prophet", "SARIMA", "Keduanya"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Forecast", use_container_width=True)

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="page-title">Forecasting Harga Saham Bank Indonesia</div>
    <div class="page-subtitle">Model: {model_choice} &nbsp;·&nbsp; Horizon: {forecast_days} hari &nbsp;·&nbsp; Saham: {selected_name}</div>
</div>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def flatten_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and normalize Date column."""
    df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
    df_raw = df_raw.reset_index()
    date_col = df_raw.columns[0]
    df_raw = df_raw.rename(columns={date_col: "Date"})
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    return df_raw


def forecast_open_prophet(df_raw: pd.DataFrame, periods: int) -> pd.Series:
    """Forecast Open price with Prophet; returns Series indexed by future dates."""
    df_open = df_raw[["Date", "Open"]].dropna().copy()
    df_open.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df_open)
    future = m.make_future_dataframe(periods=periods, freq="B")
    fc = m.predict(future).tail(periods)
    return fc.set_index("ds")["yhat"]


def forecast_open_sarima(series: pd.Series, periods: int, last_date) -> pd.Series:
    """Forecast Open price with SARIMA; returns Series indexed by future dates."""
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=periods)
    future_dates = pd.bdate_range(start=last_date, periods=periods + 1)[1:]
    return pd.Series(fc.values, index=future_dates)


def run_prophet(df_close: pd.DataFrame, hist_open: pd.Series,
                open_regressor: pd.Series, periods: int) -> pd.DataFrame:
    """
    Prophet forecast of Close price using forecasted Open as external regressor.

    Parameters
    ----------
    df_close       : DataFrame with columns ['ds', 'y'] — historical Close prices
    hist_open      : Series indexed by Timestamp — historical Open prices (same dates as df_close)
    open_regressor : Series indexed by Timestamp — forecasted Open prices (future dates)
    periods        : int — number of business days to forecast
    """
    # Build training set: merge historical Open into df_close via date index
    train = df_close.copy()
    # Normalize index of hist_open to date-only Timestamps for safe lookup
    hist_open_map = {pd.Timestamp(k).normalize(): v for k, v in hist_open.items()}
    train["open_val"] = train["ds"].map(
        lambda d: hist_open_map.get(pd.Timestamp(d).normalize(), np.nan)
    )
    train = train.dropna(subset=["open_val"])

    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.add_regressor("open_val")
    m.fit(train)

    future = m.make_future_dataframe(periods=periods, freq="B")

    # Build unified open lookup: historical + forecasted
    future_open_map = {pd.Timestamp(k).normalize(): v for k, v in open_regressor.items()}
    combined_map = {**hist_open_map, **future_open_map}   # future overwrites if overlap

    future["open_val"] = future["ds"].map(
        lambda d: combined_map.get(pd.Timestamp(d).normalize(), open_regressor.iloc[-1])
    )

    fc = m.predict(future)
    return fc.tail(periods)


def run_sarima(close_series: pd.Series, open_exog_hist: pd.Series,
               open_exog_future: pd.Series, periods: int, last_date):
    """
    SARIMAX forecast of Close price using Open as exogenous variable.
    close_series     : historical Close (aligned)
    open_exog_hist   : historical Open  (aligned, same length as close_series)
    open_exog_future : forecasted Open  (length = periods)
    """
    model = SARIMAX(
        close_series,
        exog=open_exog_hist.values.reshape(-1, 1),
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 5),
    )
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=periods, exog=open_exog_future.values.reshape(-1, 1))
    future_dates = pd.bdate_range(start=last_date, periods=periods + 1)[1:]
    return future_dates, fc


def compute_signal(forecast_close_next: float, last_close: float) -> str:
    """Simple directional signal based on 1-day-ahead forecast vs last close."""
    change = (forecast_close_next - last_close) / last_close
    if change > BUY_THRESHOLD:
        return "BUY"
    elif change < SELL_THRESHOLD:
        return "SELL"
    else:
        return "HOLD"


def signal_badge(signal: str) -> str:
    icons = {"BUY": "▲", "SELL": "▼", "HOLD": "—"}
    return f'<span class="signal-{signal}">{icons[signal]} {signal}</span>'


def color_signal(val):
    colors = {"BUY": "#065F46", "SELL": "#991B1B", "HOLD": "#92400E"}
    bg     = {"BUY": "#D1FAE5",  "SELL": "#FEE2E2",  "HOLD": "#FEF3C7"}
    c = colors.get(val, "#000")
    b = bg.get(val, "#FFF")
    return f"background-color: {b}; color: {c}; font-weight: 700;"


# ── Main ──────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Fetching data & running models…"):

        # ── Download ────────────────────────────────────────────────────────
        df_raw = yf.download(ticker, start=str(start_date), end=str(date.today()), auto_adjust=True)
        if df_raw.empty:
            st.error("Data tidak ditemukan. Coba ticker lain atau ubah tanggal.")
            st.stop()

        df_raw = flatten_df(df_raw)

        # ── Align Close & Open (drop NaN) ────────────────────────────────────
        df_close_with_open = df_raw[["Date", "Close", "Open"]].dropna().copy()
        df_close_with_open["Date"] = pd.to_datetime(df_close_with_open["Date"])

        # Prophet-style rename for Close
        df_close = df_close_with_open[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

        last_close   = df_close_with_open["Close"].iloc[-1]
        last_open    = df_close_with_open["Open"].iloc[-1]
        last_date    = df_close_with_open["Date"].iloc[-1]
        total_rows   = len(df_close_with_open)

        # ── Step 1: Forecast Open Price ───────────────────────────────────────
        if model_choice in ["Prophet", "Keduanya"]:
            open_fc_prophet = forecast_open_prophet(df_close_with_open.rename(columns={"Date": "Date"}), forecast_days)

        if model_choice in ["SARIMA", "Keduanya"]:
            open_series_hist = df_close_with_open["Open"].values
            open_fc_sarima   = forecast_open_sarima(
                pd.Series(open_series_hist, index=df_close_with_open["Date"]),
                forecast_days,
                last_date,
            )

        # ── Step 2: Forecast Close Price (with Open as regressor) ─────────────
        prophet_fc   = None
        sarima_dates = None
        sarima_fc    = None

        if model_choice in ["Prophet", "Keduanya"]:
            hist_open_series = pd.Series(
                df_close_with_open["Open"].values,
                index=pd.to_datetime(df_close_with_open["Date"]),
            )
            prophet_fc = run_prophet(df_close, hist_open_series, open_fc_prophet, forecast_days)

        if model_choice in ["SARIMA", "Keduanya"]:
            close_series_hist = pd.Series(
                df_close_with_open["Close"].values,
                index=df_close_with_open["Date"],
            )
            open_hist_aligned = pd.Series(
                df_close_with_open["Open"].values,
                index=df_close_with_open["Date"],
            )
            sarima_dates, sarima_fc = run_sarima(
                close_series_hist,
                open_hist_aligned,
                open_fc_sarima,
                forecast_days,
                last_date,
            )

        # ── Step 3: Compute Signals ───────────────────────────────────────────
        signal_prophet = None
        signal_sarima  = None

        if prophet_fc is not None:
            signal_prophet = compute_signal(float(prophet_fc["yhat"].iloc[0]), last_close)

        if sarima_fc is not None:
            signal_sarima = compute_signal(float(sarima_fc.iloc[0]), last_close)

    # ── Metrics Row ──────────────────────────────────────────────────────────
    cols = st.columns(5)
    cards = [
        ("Saham",              selected_name,          ""),
        ("Last Close Price",   f"Rp {last_close:,.0f}",""),
        ("Last Open Price",    f"Rp {last_open:,.0f}", ""),
        ("Data Points",        f"{total_rows:,}",      "hari perdagangan"),
        ("Forecast Horizon",   f"{forecast_days} hari","ke depan"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Signal Row ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Trading Signal (H+1)</div>', unsafe_allow_html=True)

    sig_cols = st.columns(4)

    if signal_prophet is not None:
        next_close_p = float(prophet_fc["yhat"].iloc[0])
        chg_p = (next_close_p - last_close) / last_close * 100
        sig_cols[0].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prophet — Signal</div>
            <div style="margin: 0.4rem 0;">{signal_badge(signal_prophet)}</div>
            <div class="metric-sub">Forecast H+1: Rp {next_close_p:,.0f} ({chg_p:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    if signal_sarima is not None:
        next_close_s = float(sarima_fc.iloc[0])
        chg_s = (next_close_s - last_close) / last_close * 100
        sig_cols[1].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SARIMA — Signal</div>
            <div style="margin: 0.4rem 0;">{signal_badge(signal_sarima)}</div>
            <div class="metric-sub">Forecast H+1: Rp {next_close_s:,.0f} ({chg_s:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> Sinyal buy/sell dihasilkan dari perbandingan forecast harga Close H+1 
        terhadap harga Close terakhir dengan threshold ±1%. Ini <em>bukan</em> rekomendasi investasi. 
        Output model bersifat prediktif dan tidak menjamin akurasi di pasar nyata.
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Price Chart & Forecast</div>', unsafe_allow_html=True)

    # Show only last 2 years of historical data in chart to keep it readable
    cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=2)
    df_chart = df_close_with_open[df_close_with_open["Date"] >= cutoff]

    fig = go.Figure()

    # Historical Close
    fig.add_trace(go.Scatter(
        x=df_chart["Date"], y=df_chart["Close"],
        name="Historical Close",
        line=dict(color="#0F4C81", width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Close: Rp %{y:,.0f}<extra></extra>",
    ))

    # Historical Open (subtle)
    fig.add_trace(go.Scatter(
        x=df_chart["Date"], y=df_chart["Open"],
        name="Historical Open",
        line=dict(color="#94A3B8", width=1, dash="dot"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Open: Rp %{y:,.0f}<extra></extra>",
    ))

    # Prophet forecast
    if prophet_fc is not None:
        fig.add_trace(go.Scatter(
            x=prophet_fc["ds"], y=prophet_fc["yhat"],
            name="Forecast Close (Prophet)",
            line=dict(color="#10B981", width=2, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Prophet: Rp %{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([prophet_fc["ds"], prophet_fc["ds"][::-1]]),
            y=pd.concat([prophet_fc["yhat_upper"], prophet_fc["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(16,185,129,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prophet CI",
            hoverinfo="skip",
        ))
        # Forecasted Open from Prophet
        open_fc_prophet_chart = open_fc_prophet[open_fc_prophet.index >= pd.Timestamp(date.today())]
        fig.add_trace(go.Scatter(
            x=open_fc_prophet_chart.index,
            y=open_fc_prophet_chart.values,
            name="Forecast Open (Prophet)",
            line=dict(color="#6EE7B7", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast Open: Rp %{y:,.0f}<extra></extra>",
        ))

    # SARIMA forecast
    if sarima_fc is not None:
        fig.add_trace(go.Scatter(
            x=sarima_dates, y=sarima_fc,
            name="Forecast Close (SARIMA)",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>SARIMA: Rp %{y:,.0f}<extra></extra>",
        ))
        # Forecasted Open from SARIMA
        fig.add_trace(go.Scatter(
            x=open_fc_sarima.index,
            y=open_fc_sarima.values,
            name="Forecast Open (SARIMA)",
            line=dict(color="#FDE68A", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast Open: Rp %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, Helvetica Neue, Arial", size=12, color="#1A1A2E"),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Price (IDR)",
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickformat=",",
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast Tables ───────────────────────────────────────────────────────
    if prophet_fc is not None:
        st.markdown('<div class="section-header">Forecast Table — Prophet</div>', unsafe_allow_html=True)

        tbl_p = prophet_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        tbl_p.columns = ["Tanggal", "Forecast Close", "Batas Bawah", "Batas Atas"]
        tbl_p["Tanggal"] = pd.to_datetime(tbl_p["Tanggal"]).dt.strftime("%Y-%m-%d")

        # Add open forecast
        open_fc_p_list = [open_fc_prophet.get(pd.Timestamp(d), np.nan) for d in tbl_p["Tanggal"]]
        tbl_p.insert(1, "Forecast Open", open_fc_p_list)

        # Add signal column
        tbl_p["Signal"] = ["BUY" if (row["Forecast Close"] - last_close) / last_close > BUY_THRESHOLD
                           else ("SELL" if (row["Forecast Close"] - last_close) / last_close < SELL_THRESHOLD else "HOLD")
                           for _, row in tbl_p.iterrows()]

        # Format numbers
        for col in ["Forecast Open", "Forecast Close", "Batas Bawah", "Batas Atas"]:
            tbl_p[col] = tbl_p[col].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "-")

        styled_p = tbl_p.style.applymap(color_signal, subset=["Signal"])
        st.dataframe(styled_p, use_container_width=True, hide_index=True)

    if sarima_fc is not None:
        st.markdown('<div class="section-header">Forecast Table — SARIMA</div>', unsafe_allow_html=True)

        sarima_close_vals = list(sarima_fc.values)
        sarima_open_vals  = [open_fc_sarima.get(d, np.nan) for d in sarima_dates]

        signals_s = []
        for v in sarima_close_vals:
            chg = (v - last_close) / last_close
            signals_s.append("BUY" if chg > BUY_THRESHOLD else ("SELL" if chg < SELL_THRESHOLD else "HOLD"))

        tbl_s = pd.DataFrame({
            "Tanggal":        [d.strftime("%Y-%m-%d") for d in sarima_dates],
            "Forecast Open":  [f"Rp {v:,.0f}" if pd.notna(v) else "-" for v in sarima_open_vals],
            "Forecast Close": [f"Rp {v:,.0f}" for v in sarima_close_vals],
            "Signal":         signals_s,
        })

        styled_s = tbl_s.style.applymap(color_signal, subset=["Signal"])
        st.dataframe(styled_s, use_container_width=True, hide_index=True)

    # ── Footer note ───────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.72rem; color:#94A3B8; text-align:right;">'
        f'Data source: Yahoo Finance &nbsp;·&nbsp; Model: {model_choice} &nbsp;·&nbsp; '
        f'Generated: {date.today().strftime("%d %b %Y")}'
        '</div>',
        unsafe_allow_html=True,
    )
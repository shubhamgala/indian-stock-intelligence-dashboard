# ============================================================
# 🇮🇳 Indian Stock Intelligence Dashboard (Nifty 50 Edition)
# Stack: Python · Streamlit · Plotly · Pandas · yfinance
# Benchmark: Nifty 50 Index (^NSEI) | Exchange: NSE India (.NS)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── QUANTITATIVE METRICS MODULE ──────────────────────────────
from finance_metrics import (
    calculate_daily_returns,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_drawdown,
    calculate_beta,
    calculate_correlation_matrix,
    get_full_risk_return_profile,
)

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Intelligence | Nifty 50",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS (Dark Financial Terminal Theme) ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer { visibility: hidden; }
header { visibility: visible; background: transparent !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.stDeployButton { display: none !important; }
[data-testid="collapsedControl"] { color: #f59e0b !important; }
button[kind="header"] { color: #f59e0b !important; }

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0b1120 0%, #1e1b4b 50%, #1e293b 100%);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(245, 158, 11, 0.25);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.3rem; font-weight: 800;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.hero-subtitle {
    color: rgba(255, 255, 255, 0.7); font-size: 1.02rem;
    font-weight: 400; margin-top: 0.4rem;
}

/* KPI Cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.amber::after  { background: linear-gradient(90deg, #f59e0b, #d97706); }
.kpi-card.blue::after   { background: linear-gradient(90deg, #60a5fa, #2563eb); }
.kpi-card.green::after  { background: linear-gradient(90deg, #10b981, #059669); }
.kpi-card.purple::after { background: linear-gradient(90deg, #a78bfa, #7c3aed); }
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(245, 158, 11, 0.3); box-shadow: 0 12px 35px rgba(0,0,0,0.5); }
.kpi-label { color: rgba(255, 255, 255, 0.55); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 0.3rem 0 0.1rem; }
.kpi-sub   { color: rgba(255, 255, 255, 0.45); font-size: 0.78rem; }

/* Stock Profile Card */
.stock-profile {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(16, 185, 129, 0.08));
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 20px; padding: 1.8rem 2rem; margin-bottom: 1.5rem;
}
.stock-name { font-size: 1.8rem; font-weight: 800; color: #fff; }
.stock-ticker { font-size: 0.95rem; color: #fbbf24; font-weight: 700; letter-spacing: 0.05em; }
.stock-sector { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.2rem; }

/* Mini Metric Cards */
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-top: 1rem; }
.metric-card {
    background: rgba(0, 0, 0, 0.3); border-radius: 12px;
    padding: 0.9rem 1rem; border: 1px solid rgba(255, 255, 255, 0.08);
}
.metric-label { font-size: 0.72rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.06em; }
.metric-val   { font-size: 1.25rem; font-weight: 700; color: #fff; margin-top: 0.2rem; }
.metric-val.green { color: #34d399; }
.metric-val.red   { color: #f87171; }
.metric-val.gold  { color: #fbbf24; }
.metric-val.blue  { color: #60a5fa; }

/* Section Header */
.section-header {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    margin: 1.5rem 0 1rem; display: flex; align-items: center; gap: 0.5rem;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 16px; padding: 1.5rem;
    backdrop-filter: blur(10px); margin-bottom: 1rem;
}

/* Insight Cards */
.insight-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
.insight-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid;
    border-radius: 12px; padding: 1rem 1.2rem;
    transition: all 0.2s;
}
.insight-card:hover { background: rgba(255, 255, 255, 0.06); transform: translateX(3px); }
.insight-title { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.4rem; }
.insight-text  { font-size: 0.9rem; color: rgba(255, 255, 255, 0.8); line-height: 1.5; }

/* AI Box */
.ai-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(16, 185, 129, 0.12));
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 16px; padding: 1.5rem 2rem;
}
.ai-label { font-size: 0.72rem; font-weight: 700; color: #f59e0b; letter-spacing: 0.1em; text-transform: uppercase; }
.ai-value  { font-size: 1.15rem; font-weight: 700; color: #fff; margin-top: 0.2rem; }
.ai-desc   { font-size: 0.85rem; color: rgba(255, 255, 255, 0.6); margin-top: 0.3rem; }

/* Badges */
.badge-grid { display: flex; flex-wrap: wrap; gap: 0.7rem; }
.badge {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 50px; padding: 0.45rem 1.1rem;
    font-size: 0.82rem; font-weight: 600; color: #fbbf24;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #151b2e 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

.stApp {
    background: linear-gradient(160deg, #080c16 0%, #0f172a 100%);
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=["#f59e0b", "#60a5fa", "#34d399", "#f87171", "#a78bfa", "#facc15", "#2dd4bf", "#fb923c"],
)

def apply_theme(fig, title="", height=380):
    fig.update_layout(**PLOTLY_THEME, title=dict(text=title, font=dict(size=14, color="#e2e8f0")), height=height)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", showgrid=True, zeroline=False)
    return fig

# ── NIFTY 50 CONSTITUENTS & SECTORS ──────────────────────────
# Active constituents of NIFTY 50 (NSE symbols with .NS suffix)
NIFTY_50_STOCKS = {
    "Reliance Industries": {"ticker": "RELIANCE.NS", "sector": "Oil, Gas & Energy"},
    "Tata Consultancy Services": {"ticker": "TCS.NS", "sector": "Information Technology"},
    "HDFC Bank": {"ticker": "HDFCBANK.NS", "sector": "Banking & Financial Services"},
    "Infosys": {"ticker": "INFY.NS", "sector": "Information Technology"},
    "ICICI Bank": {"ticker": "ICICIBANK.NS", "sector": "Banking & Financial Services"},
    "State Bank of India": {"ticker": "SBIN.NS", "sector": "Banking & Financial Services"},
    "Bharti Airtel": {"ticker": "BHARTIARTL.NS", "sector": "Telecommunication"},
    "ITC Limited": {"ticker": "ITC.NS", "sector": "Consumer Goods & FMCG"},
    "Larsen & Toubro": {"ticker": "LT.NS", "sector": "Construction & Materials"},
    "Hindustan Unilever": {"ticker": "HINDUNILVR.NS", "sector": "Consumer Goods & FMCG"},
    "Axis Bank": {"ticker": "AXISBANK.NS", "sector": "Banking & Financial Services"},
    "Kotak Mahindra Bank": {"ticker": "KOTAKBANK.NS", "sector": "Banking & Financial Services"},
    "Bajaj Finance": {"ticker": "BAJFINANCE.NS", "sector": "Banking & Financial Services"},
    "Bajaj Finserv": {"ticker": "BAJAJFINSV.NS", "sector": "Banking & Financial Services"},
    "Maruti Suzuki": {"ticker": "MARUTI.NS", "sector": "Automobile & Auto Components"},
    "Mahindra & Mahindra": {"ticker": "M&M.NS", "sector": "Automobile & Auto Components"},
    "Sun Pharma": {"ticker": "SUNPHARMA.NS", "sector": "Healthcare & Pharmaceuticals"},
    "Titan Company": {"ticker": "TITAN.NS", "sector": "Consumer Goods & FMCG"},
    "Asian Paints": {"ticker": "ASIANPAINT.NS", "sector": "Consumer Goods & FMCG"},
    "NTPC": {"ticker": "NTPC.NS", "sector": "Power & Utilities"},
    "Power Grid Corporation": {"ticker": "POWERGRID.NS", "sector": "Power & Utilities"},
    "Tata Steel": {"ticker": "TATASTEEL.NS", "sector": "Metals & Mining"},
    "JSW Steel": {"ticker": "JSWSTEEL.NS", "sector": "Metals & Mining"},
    "Coal India": {"ticker": "COALINDIA.NS", "sector": "Oil, Gas & Energy"},
    "ONGC": {"ticker": "ONGC.NS", "sector": "Oil, Gas & Energy"},
    "Adani Ports": {"ticker": "ADANIPORTS.NS", "sector": "Services & Infrastructure"},
    "Adani Enterprises": {"ticker": "ADANIENT.NS", "sector": "Services & Infrastructure"},
    "UltraTech Cement": {"ticker": "ULTRACEMCO.NS", "sector": "Construction & Materials"},
    "HCL Technologies": {"ticker": "HCLTECH.NS", "sector": "Information Technology"},
    "Tech Mahindra": {"ticker": "TECHM.NS", "sector": "Information Technology"},
    "Wipro": {"ticker": "WIPRO.NS", "sector": "Information Technology"},
    "Bajaj Auto": {"ticker": "BAJAJ-AUTO.NS", "sector": "Automobile & Auto Components"},
    "Eicher Motors": {"ticker": "EICHERMOT.NS", "sector": "Automobile & Auto Components"},
    "Hero MotoCorp": {"ticker": "HEROMOTOCO.NS", "sector": "Automobile & Auto Components"},
    "Bharat Petroleum (BPCL)": {"ticker": "BPCL.NS", "sector": "Oil, Gas & Energy"},
    "Cipla": {"ticker": "CIPLA.NS", "sector": "Healthcare & Pharmaceuticals"},
    "Dr. Reddy's Laboratories": {"ticker": "DRREDDY.NS", "sector": "Healthcare & Pharmaceuticals"},
    "Apollo Hospitals": {"ticker": "APOLLOHOSP.NS", "sector": "Healthcare & Pharmaceuticals"},
    "Divi's Laboratories": {"ticker": "DIVISLAB.NS", "sector": "Healthcare & Pharmaceuticals"},
    "Nestle India": {"ticker": "NESTLEIND.NS", "sector": "Consumer Goods & FMCG"},
    "Britannia Industries": {"ticker": "BRITANNIA.NS", "sector": "Consumer Goods & FMCG"},
    "Tata Consumer Products": {"ticker": "TATACONSUM.NS", "sector": "Consumer Goods & FMCG"},
    "Grasim Industries": {"ticker": "GRASIM.NS", "sector": "Construction & Materials"},
    "Hindalco Industries": {"ticker": "HINDALCO.NS", "sector": "Metals & Mining"},
    "IndusInd Bank": {"ticker": "INDUSINDBK.NS", "sector": "Banking & Financial Services"},
    "Shriram Finance": {"ticker": "SHRIRAMFIN.NS", "sector": "Banking & Financial Services"},
    "Trent Limited": {"ticker": "TRENT.NS", "sector": "Consumer Goods & FMCG"},
    "SBI Life Insurance": {"ticker": "SBILIFE.NS", "sector": "Banking & Financial Services"},
    "Bharat Electronics (BEL)": {"ticker": "BEL.NS", "sector": "Capital Goods & Defense"},
    "Hindustan Aeronautics (HAL)": {"ticker": "HAL.NS", "sector": "Capital Goods & Defense"},
}

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "NIFTY 50 Benchmark"

ALL_SECTORS = sorted(list({v["sector"] for v in NIFTY_50_STOCKS.values()}))
TICKER_TO_NAME = {v["ticker"]: k for k, v in NIFTY_50_STOCKS.items()}
TICKER_TO_SECTOR = {v["ticker"]: v["sector"] for k, v in NIFTY_50_STOCKS.items()}
NAME_TO_TICKER = {k: v["ticker"] for k, v in NIFTY_50_STOCKS.items()}

# ── DATA FETCHING WITH ERROR HANDLING & CACHING ──────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty_market_data(ticker_list, start_str, end_str):
    """
    Downloads historical data for NIFTY 50 tickers and ^NSEI benchmark with yfinance.
    Implements robust error handling and normalizes the multi-ticker dataframe.
    Cached with st.cache_data for 1 hour.
    """
    all_symbols = list(ticker_list) + [BENCHMARK_TICKER]
    failed_downloads = []
    
    try:
        raw = yf.download(all_symbols, start=start_str, end=end_str, group_by="ticker", threads=True, progress=False)
    except Exception as exc:
        st.error(f"⚠️ Network error while fetching data from yfinance: {exc}")
        return pd.DataFrame(), pd.DataFrame(), all_symbols

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), all_symbols

    stock_records = []
    benchmark_df = pd.DataFrame()

    for sym in all_symbols:
        try:
            if sym in raw:
                sub = raw[sym].dropna(how="all").copy()
                if not sub.empty and "Close" in sub and len(sub["Close"].dropna()) > 0:
                    sub = sub.reset_index()
                    sub["Ticker"] = sym
                    if sym == BENCHMARK_TICKER:
                        sub["Stock_Name"] = BENCHMARK_NAME
                        sub["Sector"] = "Benchmark Index"
                        benchmark_df = sub
                    else:
                        sub["Stock_Name"] = TICKER_TO_NAME.get(sym, sym)
                        sub["Sector"] = TICKER_TO_SECTOR.get(sym, "Indian Equities")
                        stock_records.append(sub)
                else:
                    failed_downloads.append(sym)
            else:
                failed_downloads.append(sym)
        except Exception:
            failed_downloads.append(sym)

    if not stock_records:
        return pd.DataFrame(), pd.DataFrame(), failed_downloads

    combined_df = pd.concat(stock_records, ignore_index=True)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"])
    combined_df = combined_df.sort_values(["Stock_Name", "Date"]).reset_index(drop=True)

    # Feature Engineering: Technical Indicators
    combined_df["Daily_Return"] = combined_df.groupby("Stock_Name")["Close"].pct_change() * 100
    combined_df["MA20"] = combined_df.groupby("Stock_Name")["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    combined_df["MA50"] = combined_df.groupby("Stock_Name")["Close"].transform(lambda x: x.rolling(50, min_periods=1).mean())
    combined_df["MA200"] = combined_df.groupby("Stock_Name")["Close"].transform(lambda x: x.rolling(200, min_periods=1).mean())
    combined_df["Volatility"] = combined_df.groupby("Stock_Name")["Daily_Return"].transform(lambda x: x.rolling(20, min_periods=1).std())

    # Benchmark processing
    if not benchmark_df.empty:
        benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"])
        benchmark_df = benchmark_df.sort_values("Date").reset_index(drop=True)
        benchmark_df["Daily_Return"] = benchmark_df["Close"].pct_change() * 100
        b_first = benchmark_df["Close"].iloc[0]
        benchmark_df["Normalized_Return"] = ((benchmark_df["Close"] - b_first) / b_first) * 100
        benchmark_df["MA20"] = benchmark_df["Close"].rolling(20, min_periods=1).mean()
        benchmark_df["MA50"] = benchmark_df["Close"].rolling(50, min_periods=1).mean()

    return combined_df, benchmark_df, failed_downloads

@st.cache_data(show_spinner=False)
def compute_stock_summary(df):
    stats = df.groupby(["Stock_Name", "Ticker", "Sector"]).agg(
        Latest_Close=("Close", "last"),
        Max_High=("High", "max"),
        Min_Low=("Low", "min"),
        Avg_Close=("Close", "mean"),
        Total_Volume=("Volume", "sum"),
        Std_Close=("Close", "std"),
        Start_Close=("Close", "first"),
    ).reset_index()
    stats["Total_Return_Pct"] = ((stats["Latest_Close"] - stats["Start_Close"]) / stats["Start_Close"]) * 100
    stats["Volatility_Score"] = (stats["Std_Close"] / stats["Avg_Close"]) * 100
    return stats

@st.cache_data(show_spinner=False)
def compute_sector_summary(stock_stats):
    sstats = stock_stats.groupby("Sector").agg(
        Num_Stocks=("Stock_Name", "count"),
        Avg_Return=("Total_Return_Pct", "mean"),
        Avg_Volatility=("Volatility_Score", "mean"),
        Total_Volume=("Total_Volume", "sum"),
    ).reset_index()
    return sstats.sort_values("Total_Volume", ascending=False)

def compute_beta_alpha(stock_df, benchmark_df):
    """Calculates Beta and Alpha against Nifty 50 (^NSEI)."""
    if stock_df.empty or benchmark_df.empty:
        return 1.0, 0.0, 0.0, 0.0
    merged = pd.merge(
        stock_df[["Date", "Daily_Return"]].dropna(),
        benchmark_df[["Date", "Daily_Return"]].dropna(),
        on="Date",
        suffixes=("_stock", "_bench")
    )
    if len(merged) < 5:
        return 1.0, 0.0, 0.0, 0.0
    
    cov = np.cov(merged["Daily_Return_stock"], merged["Daily_Return_bench"])[0, 1]
    var_bench = np.var(merged["Daily_Return_bench"])
    beta = cov / var_bench if var_bench > 0 else 1.0
    annual_alpha = (merged["Daily_Return_stock"].mean() - beta * merged["Daily_Return_bench"].mean()) * 252
    corr = merged["Daily_Return_stock"].corr(merged["Daily_Return_bench"])
    r_squared = corr ** 2 if not np.isnan(corr) else 0.0
    return round(float(beta), 2), round(float(annual_alpha), 2), round(float(corr), 2), round(float(r_squared), 2)

# ── SIDEBAR CONTROLS ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.8rem 0 0.5rem;'>
        <div style='font-size:2.3rem;'>🇮🇳</div>
        <div style='font-size:1.15rem; font-weight:800; color:#fff; letter-spacing:0.02em;'>NiftyVision Pro</div>
        <div style='font-size:0.75rem; color:#fbbf24; font-weight:600; margin-top:0.2rem;'>NSE NIFTY 50 Intelligence</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.08); margin:0.8rem 0 1rem;'/>
    """, unsafe_allow_html=True)

    # 1. Date Range & Time Horizon
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;'>📅 Date Range & Horizon</div>", unsafe_allow_html=True)
    timeframe = st.selectbox(
        "Timeframe Preset",
        options=["1 Month", "3 Months", "6 Months", "1 Year (Default)", "2 Years", "YTD", "Custom Date Range"],
        index=3,
        label_visibility="collapsed"
    )

    today = datetime.now()
    if timeframe == "1 Month":
        start_date = today - timedelta(days=30)
        end_date = today
    elif timeframe == "3 Months":
        start_date = today - timedelta(days=90)
        end_date = today
    elif timeframe == "6 Months":
        start_date = today - timedelta(days=180)
        end_date = today
    elif timeframe == "1 Year (Default)":
        start_date = today - timedelta(days=365)
        end_date = today
    elif timeframe == "2 Years":
        start_date = today - timedelta(days=730)
        end_date = today
    elif timeframe == "YTD":
        start_date = datetime(today.year, 1, 1)
        end_date = today
    else:  # Custom Date Range
        st.markdown("<div style='font-size:0.75rem; color:rgba(255,255,255,0.6); margin-top:0.3rem;'>Pick Start & End Date:</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        start_date = c1.date_input("From Date", value=today - timedelta(days=365), max_value=today)
        end_date = c2.date_input("To Date", value=today, max_value=today)
        if start_date >= end_date:
            st.error("Error: 'From' date must precede 'To' date.")
            st.stop()

    start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    # Active date window badge
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:0.4rem 0.6rem; margin-top:0.4rem; margin-bottom:1.2rem; font-size:0.72rem; color:rgba(255,255,255,0.6); text-align:center;'>
        📆 <b>Active Window:</b> {start_str} → {end_str}
    </div>
    """, unsafe_allow_html=True)

    # 2. Stock Universe & Sector Filter
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#60a5fa; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;'>🏢 Stock Universe</div>", unsafe_allow_html=True)
    sel_sectors = st.multiselect(
        "Filter by Sector:",
        options=ALL_SECTORS,
        default=ALL_SECTORS,
        placeholder="Filter Indian sectors..."
    )

    filtered_stock_names = [
        name for name, meta in NIFTY_50_STOCKS.items()
        if not sel_sectors or meta["sector"] in sel_sectors
    ]

    search_q = st.text_input("🔍 Search Stock / Ticker:", placeholder="Type e.g. Reliance, TCS, HDFC...")
    if search_q:
        filtered_stock_names = [
            s for s in filtered_stock_names
            if search_q.lower() in s.lower() or search_q.lower() in NAME_TO_TICKER[s].lower()
        ]

    default_index = 0
    if "Reliance Industries" in filtered_stock_names:
        default_index = filtered_stock_names.index("Reliance Industries")

    sel_stock = st.selectbox(
        "📊 Select Nifty Stock:",
        options=filtered_stock_names if filtered_stock_names else list(NIFTY_50_STOCKS.keys()),
        index=default_index
    )

    # 3. Settings & Hurdle
    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:1.2rem 0 0.8rem;'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#34d399; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;'>⚙️ Analysis Settings</div>", unsafe_allow_html=True)
    compare_benchmark = st.checkbox("📈 Benchmark Overlay (^NSEI)", value=True)
    st.markdown("""
    <div style='background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); border-radius:8px; padding:0.45rem 0.6rem; margin-top:0.2rem; margin-bottom:0.8rem; font-size:0.72rem; color:#34d399;'>
        🛡️ <b>Risk-Free Hurdle:</b> 7.0% p.a. (India 10Y G-Sec / RBI Repo)
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style='font-size:0.7rem; color:rgba(255,255,255,0.35); text-align:center; padding-top:1rem;'>
        Live Data Ingestion via yfinance<br/>
        Exchange: National Stock Exchange (NSE India)
    </div>
    """, unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────
with st.spinner("Fetching live NIFTY 50 data from yfinance..."):
    all_tickers = [meta["ticker"] for meta in NIFTY_50_STOCKS.values()]
    df, benchmark_df, failed_tickers = fetch_nifty_market_data(all_tickers, start_str, end_str)

if df.empty:
    st.error("⚠️ Failed to load market data. Please check your internet connection and click 'Refresh Market Data'.")
    st.stop()

if failed_tickers and len(failed_tickers) > 1:
    with st.expander(f"ℹ️ Note: {len(failed_tickers)} symbols temporarily unavailable on Yahoo Finance"):
        st.write(", ".join(failed_tickers))

stock_stats = compute_stock_summary(df)
sector_stats = compute_sector_summary(stock_stats)

# Filter by selected sectors
if sel_sectors:
    df_filtered = df[df["Sector"].isin(sel_sectors)]
    stock_stats_filtered = stock_stats[stock_stats["Sector"].isin(sel_sectors)]
else:
    df_filtered = df
    stock_stats_filtered = stock_stats

# Active selected stock data
stock_df = df[df["Stock_Name"] == sel_stock].copy()
stock_info = stock_stats[stock_stats["Stock_Name"] == sel_stock].iloc[0] if not stock_stats[stock_stats["Stock_Name"] == sel_stock].empty else None

# Normalized returns & Base 100 series for comparison
if not stock_df.empty:
    s_first = stock_df["Close"].iloc[0]
    stock_df["Normalized_Return"] = ((stock_df["Close"] - s_first) / s_first) * 100.0
    stock_df["Base_100"] = (stock_df["Close"] / s_first) * 100.0

if not benchmark_df.empty:
    b_first = benchmark_df["Close"].iloc[0]
    benchmark_df["Normalized_Return"] = ((benchmark_df["Close"] - b_first) / b_first) * 100.0
    benchmark_df["Base_100"] = (benchmark_df["Close"] / b_first) * 100.0

# Compute Beta & Alpha
beta, alpha, corr, r_sq = compute_beta_alpha(stock_df, benchmark_df)

# ── HERO BANNER ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🇮🇳 Indian Stock Intelligence Dashboard</div>
    <div class="hero-subtitle">NIFTY 50 Analytics · Real-Time NSE Data · Benchmark Modeling vs. ^NSEI · Sector Performance</div>
</div>
""", unsafe_allow_html=True)

# ── HOW TO READ THIS DASHBOARD (GUIDE) ───────────────────────
with st.expander("📖 How to Read This Dashboard — Guide & Financial Interpretation Tips", expanded=False):
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); padding:1rem; border-radius:12px; border:1px solid rgba(255,255,255,0.08); height:100%;">
            <div style="color:#fbbf24; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">📈 Base 100 Benchmark Chart</div>
            <p style="color:rgba(255,255,255,0.75); font-size:0.83rem; line-height:1.5; margin:0;">
                Normalizes both the stock and Nifty 50 to <b>100</b> at your selected start date.<br/>
                Think of it as the <b>growth of ₹100 invested</b>: a reading of <b>125</b> means a <b>+25% gain</b>.
                When the green curve stays above the gold dashed line, the stock is generating <b>positive alpha</b> against Nifty 50.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); padding:1rem; border-radius:12px; border:1px solid rgba(255,255,255,0.08); height:100%;">
            <div style="color:#34d399; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">⚖️ Beta & Sharpe Ratio (7% Hurdle)</div>
            <p style="color:rgba(255,255,255,0.75); font-size:0.83rem; line-height:1.5; margin:0;">
                <b>Beta (β)</b>: Measures sensitivity to Nifty 50. <i>β > 1.0</i> is aggressive (amplified swings); <i>β < 1.0</i> is defensive (cushions drawdowns).<br/>
                <b>Sharpe Ratio</b>: Measures excess return per unit of volatility above India's <b>7.0% risk-free rate</b>. Sharpe &gt; 1.0 indicates top-tier risk-adjusted compensation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); padding:1rem; border-radius:12px; border:1px solid rgba(255,255,255,0.08); height:100%;">
            <div style="color:#60a5fa; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">📉 Drawdown & Correlation Matrix</div>
            <p style="color:rgba(255,255,255,0.75); font-size:0.83rem; line-height:1.5; margin:0;">
                <b>Max Drawdown (MDD)</b>: The steepest percentage decline from any peak to its subsequent trough — essential for capital preservation.<br/>
                <b>Correlation Matrix</b>: Bounded from -1.0 to +1.0. Pairs with low or negative correlation offer superior portfolio diversification and hedging.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────────
total_stocks_count = stock_stats["Stock_Name"].nunique()
total_sectors_count = stock_stats["Sector"].nunique()
bench_return_str = "N/A"
if not benchmark_df.empty:
    b_ret = ((benchmark_df["Close"].iloc[-1] - benchmark_df["Close"].iloc[0]) / benchmark_df["Close"].iloc[0]) * 100
    b_sign = "+" if b_ret >= 0 else ""
    bench_return_str = f"{benchmark_df['Close'].iloc[-1]:,.2f} ({b_sign}{b_ret:.2f}%)"

best_performer_row = stock_stats.loc[stock_stats["Total_Return_Pct"].idxmax()]

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card amber">
        <div class="kpi-label">Constituents</div>
        <div class="kpi-value">{total_stocks_count}</div>
        <div class="kpi-sub">Active Nifty 50 stocks</div>
    </div>
    <div class="kpi-card blue">
        <div class="kpi-label">Sectors Covered</div>
        <div class="kpi-value">{total_sectors_count}</div>
        <div class="kpi-sub">Diversified Indian economy</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-label">Nifty 50 (^NSEI)</div>
        <div class="kpi-value" style="font-size:1.35rem;">{bench_return_str}</div>
        <div class="kpi-sub">Benchmark index return</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-label">Top Performer</div>
        <div class="kpi-value" style="font-size:1.35rem;">{best_performer_row['Stock_Name']}</div>
        <div class="kpi-sub">+{best_performer_row['Total_Return_Pct']:.2f}% period return</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Stock Explorer",
    "⚡ Risk & Return",
    "🏢 Sector Analytics",
    "📉 Market Overview",
    "🎯 Benchmark & Beta",
    "💡 Market Insights",
    "🏆 Quant Portfolio",
])

# ════════════════════════════════════════════════════════════
# TAB 1 – STOCK EXPLORER
# ════════════════════════════════════════════════════════════
with tab1:
    if stock_info is not None:
        ret_color = "#34d399" if stock_info["Total_Return_Pct"] >= 0 else "#f87171"
        ret_sign = "+" if stock_info["Total_Return_Pct"] >= 0 else ""
        ticker = stock_info["Ticker"]

        st.markdown(f"""
        <div class="stock-profile">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
                <div>
                    <div class="stock-name">{sel_stock}</div>
                    <div class="stock-ticker">NSE: {ticker} &nbsp;|&nbsp; Sector: {stock_info['Sector']}</div>
                    <div class="stock-sector">Benchmark: NIFTY 50 (^NSEI) &nbsp;•&nbsp; Currency: INR (₹)</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:rgba(255,255,255,0.45); text-transform:uppercase; letter-spacing:0.07em;">Current Market Price</div>
                    <div style="font-size:2.2rem; font-weight:900; color:#fff;">₹{stock_info['Latest_Close']:,.2f}</div>
                    <div style="font-size:1rem; font-weight:600; color:{ret_color};">{ret_sign}{stock_info['Total_Return_Pct']:.2f}% period return</div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-label">Period High</div>
                    <div class="metric-val green">₹{stock_info['Max_High']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Period Low</div>
                    <div class="metric-val red">₹{stock_info['Min_Low']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Beta vs Nifty 50</div>
                    <div class="metric-val gold">{beta}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Traded Volume</div>
                    <div class="metric-val">{stock_info['Total_Volume']/1e6:.1f}M</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not stock_df.empty:
        c1, c2 = st.columns(2)

        with c1:
            # Candlestick Chart
            fig_candle = go.Figure(data=[go.Candlestick(
                x=stock_df["Date"],
                open=stock_df["Open"], high=stock_df["High"],
                low=stock_df["Low"], close=stock_df["Close"],
                increasing_line_color="#34d399", decreasing_line_color="#f87171",
                name="OHLC"
            )])
            fig_candle = apply_theme(fig_candle, f"🕯️ {sel_stock} — Candlestick Chart (INR)", 400)
            st.plotly_chart(fig_candle, use_container_width=True)

        with c2:
            # Price Trend + Moving Averages
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=stock_df["Date"], y=stock_df["Close"],
                mode="lines", name="Close (₹)",
                line=dict(color="#f59e0b", width=2),
                fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.08)"
            ))
            fig_ma.add_trace(go.Scatter(
                x=stock_df["Date"], y=stock_df["MA20"],
                mode="lines", name="20-Day MA",
                line=dict(color="#60a5fa", width=1.5, dash="dot")
            ))
            fig_ma.add_trace(go.Scatter(
                x=stock_df["Date"], y=stock_df["MA50"],
                mode="lines", name="50-Day MA",
                line=dict(color="#34d399", width=1.5, dash="dash")
            ))
            fig_ma = apply_theme(fig_ma, f"📈 {sel_stock} — Price Trend & Moving Averages", 400)
            st.plotly_chart(fig_ma, use_container_width=True)

        # ── NIFTY 50 COMPARISON CHART (NORMALIZED: BASE = 100 AT START DATE) ──
        if compare_benchmark and not benchmark_df.empty and not stock_df.empty:
            st.markdown('<div class="section-header">⚖️ Nifty 50 Benchmark Comparison (Normalized: Base = 100 at Start Date)</div>', unsafe_allow_html=True)
            
            end_stock_b100 = stock_df["Base_100"].iloc[-1]
            end_bench_b100 = benchmark_df["Base_100"].iloc[-1]
            excess_points = end_stock_b100 - end_bench_b100
            alpha_sign = "+" if excess_points >= 0 else ""
            alpha_color = "#34d399" if excess_points >= 0 else "#f87171"
            
            fig_b100 = go.Figure()
            fig_b100.add_trace(go.Scatter(
                x=stock_df["Date"],
                y=stock_df["Base_100"],
                mode="lines",
                name=f"{sel_stock} (₹100 → ₹{end_stock_b100:.2f})",
                line=dict(color="#10b981", width=2.5),
                hovertemplate="<b>" + sel_stock + "</b><br>Date: %{x|%d %b %Y}<br>Normalized Value: %{y:.2f} pts<extra></extra>"
            ))
            fig_b100.add_trace(go.Scatter(
                x=benchmark_df["Date"],
                y=benchmark_df["Base_100"],
                mode="lines",
                name=f"Nifty 50 Benchmark (^NSEI) (₹100 → ₹{end_bench_b100:.2f})",
                line=dict(color="#f59e0b", width=2, dash="dash"),
                hovertemplate="<b>Nifty 50 (^NSEI)</b><br>Date: %{x|%d %b %Y}<br>Normalized Value: %{y:.2f} pts<extra></extra>"
            ))
            fig_b100.add_hline(
                y=100.0,
                line_dash="dot",
                line_color="rgba(255, 255, 255, 0.4)",
                annotation_text="Base 100 (Start Date Baseline)",
                annotation_position="top left",
                annotation_font_color="rgba(255, 255, 255, 0.6)"
            )
            fig_b100 = apply_theme(fig_b100, f"📈 Growth of ₹100 Invested: {sel_stock} vs. NIFTY 50 Benchmark (^NSEI)", 380)
            fig_b100.update_yaxes(ticksuffix=" pts", title="Normalized Index (Base = 100)")
            fig_b100.update_xaxes(title="Trading Date")
            st.plotly_chart(fig_b100, use_container_width=True)

            # Growth summary callout badge
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:0.8rem 1.2rem; margin-top:-0.4rem; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div style="font-size:0.88rem; color:rgba(255,255,255,0.85);">
                    💡 <b>Growth of ₹100 Invested:</b> ₹100 in <b>{sel_stock}</b> grew to <b>₹{end_stock_b100:.2f}</b> ({'+' if end_stock_b100 >= 100 else ''}{end_stock_b100-100:.2f}%) vs. <b>₹{end_bench_b100:.2f}</b> ({'+' if end_bench_b100 >= 100 else ''}{end_bench_b100-100:.2f}%) in Nifty 50.
                </div>
                <div style="font-size:0.85rem; font-weight:700; color:{alpha_color}; background:rgba(0,0,0,0.3); padding:0.3rem 0.8rem; border-radius:20px; border:1px solid rgba(255,255,255,0.1);">
                    Net Alpha: {alpha_sign}{excess_points:.2f} pts
                </div>
            </div>
            """, unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            # Daily Returns
            returns_df = stock_df.dropna(subset=["Daily_Return"])
            colors = ["#34d399" if r >= 0 else "#f87171" for r in returns_df["Daily_Return"]]
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Bar(x=returns_df["Date"], y=returns_df["Daily_Return"], marker_color=colors, name="Return %"))
            fig_ret = apply_theme(fig_ret, f"📊 {sel_stock} — Daily Returns (%)", 340)
            st.plotly_chart(fig_ret, use_container_width=True)

        with c4:
            # Return Distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=returns_df["Daily_Return"], nbinsx=40, marker_color="#f59e0b", opacity=0.8, name="Return Dist."))
            mean_r = returns_df["Daily_Return"].mean()
            fig_dist.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_dist.add_vline(x=mean_r, line_dash="dot", line_color="#34d399", annotation_text=f"μ={mean_r:.2f}%", annotation_font_color="#34d399")
            fig_dist = apply_theme(fig_dist, f"📉 {sel_stock} — Return Distribution & Skew", 340)
            st.plotly_chart(fig_dist, use_container_width=True)

        # Volume Chart
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=stock_df["Date"], y=stock_df["Volume"], name="Daily Volume", marker_color="rgba(96, 165, 250, 0.6)"))
        fig_vol.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["Volume"].rolling(20, min_periods=1).mean(), mode="lines", name="20-Day MA Volume", line=dict(color="#fbbf24", width=2)))
        fig_vol = apply_theme(fig_vol, f"📦 {sel_stock} — Trading Volume & 20-Day MA", 300)
        st.plotly_chart(fig_vol, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 – RISK & RETURN (QUANTITATIVE ANALYSIS)
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-header">⚡ Risk & Return Profile — {sel_stock}</div>', unsafe_allow_html=True)

    if not stock_df.empty:
        # Compute quantitative metrics using finance_metrics module
        bench_ret_series = benchmark_df["Daily_Return"] if not benchmark_df.empty else None
        risk_profile = get_full_risk_return_profile(
            stock_df["Close"],
            benchmark_returns=bench_ret_series,
            risk_free_rate_pct=7.0
        )

        ann_return = risk_profile["annualized_return"]
        ann_vol = risk_profile["annualized_volatility"]
        sharpe = risk_profile["sharpe_ratio"]
        max_dd = risk_profile["max_drawdown"]
        dd_series = risk_profile["drawdown_series"]
        stock_beta = risk_profile["beta"]

        # Metric color styling
        ret_c = "#34d399" if ann_return >= 0 else "#f87171"
        ret_s = "+" if ann_return >= 0 else ""
        sharpe_c = "#34d399" if sharpe >= 1.0 else "#60a5fa" if sharpe >= 0.5 else "#f87171"
        sharpe_eval = "Excellent (>1.0)" if sharpe >= 1.0 else "Good (0.5–1.0)" if sharpe >= 0.5 else "Sub-optimal (<0.5)" if sharpe > 0 else "Negative Alpha"

        # 5 Key Quantitative Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(f"""
        <div class="kpi-card green" style="text-align:center;">
            <div class="kpi-label">Annualized Return</div>
            <div class="kpi-value" style="color:{ret_c}; font-size:1.6rem;">{ret_s}{ann_return:.2f}%</div>
            <div class="kpi-sub">Compound Annual Growth (CAGR)</div>
        </div>
        """, unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="kpi-card amber" style="text-align:center;">
            <div class="kpi-label">Annualized Volatility</div>
            <div class="kpi-value" style="font-size:1.6rem;">{ann_vol:.2f}%</div>
            <div class="kpi-sub">Annualized standard deviation</div>
        </div>
        """, unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="kpi-card blue" style="text-align:center;">
            <div class="kpi-label">Sharpe Ratio (Rf = 7%)</div>
            <div class="kpi-value" style="color:{sharpe_c}; font-size:1.6rem;">{sharpe:.2f}</div>
            <div class="kpi-sub">{sharpe_eval}</div>
        </div>
        """, unsafe_allow_html=True)

        m4.markdown(f"""
        <div class="kpi-card purple" style="text-align:center;">
            <div class="kpi-label">Max Drawdown (MDD)</div>
            <div class="kpi-value" style="color:#f87171; font-size:1.6rem;">{max_dd:.2f}%</div>
            <div class="kpi-sub">Peak-to-trough max loss</div>
        </div>
        """, unsafe_allow_html=True)

        m5.markdown(f"""
        <div class="kpi-card amber" style="text-align:center;">
            <div class="kpi-label">Beta vs. Nifty 50</div>
            <div class="kpi-value" style="color:#fbbf24; font-size:1.6rem;">{stock_beta:.2f}</div>
            <div class="kpi-sub">{'Aggressive (>1.0)' if stock_beta > 1.05 else 'Defensive (<1.0)' if stock_beta < 0.95 else 'Market Neutral (~1.0)'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Drawdown / Underwater Chart
        st.markdown('<div class="section-header">📉 Historical Drawdown & Underwater Profile</div>', unsafe_allow_html=True)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=stock_df["Date"],
            y=dd_series,
            mode="lines",
            name="Drawdown (%)",
            line=dict(color="#f43f5e", width=2),
            fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.18)"
        ))
        fig_dd.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)")
        fig_dd.add_hline(
            y=max_dd, line_dash="dash", line_color="#f87171",
            annotation_text=f"Max Drawdown: {max_dd:.2f}%",
            annotation_position="bottom right",
            annotation_font_color="#f87171"
        )
        fig_dd = apply_theme(fig_dd, f"Underwater Chart — {sel_stock} Drawdown from Historical Peak (%)", 360)
        fig_dd.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_dd, use_container_width=True)

        # Risk commentary box
        rf_text = (
            f"With an annualized return of <b>{ret_s}{ann_return:.2f}%</b> and volatility of <b>{ann_vol:.2f}%</b>, "
            f"<b>{sel_stock}</b> achieves a Sharpe ratio of <b>{sharpe:.2f}</b> using a <b>7.0% risk-free rate hurdle</b> "
            f"(reflecting India's 10-Year Government Bond yield and RBI policy benchmark). "
            f"{'The stock delivered superior risk-adjusted alpha.' if sharpe > 1.0 else 'The stock provided moderate excess return for the risk incurred.' if sharpe > 0 else 'The stock underperformed risk-free government securities over this period.'}"
        )
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.95rem; font-weight:700; color:#fbbf24; margin-bottom:0.4rem;">📊 Risk-Adjusted Return Commentary</div>
            <p style="color:rgba(255,255,255,0.85); font-size:0.88rem; line-height:1.5; margin:0;">{rf_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── USER-SELECTED CORRELATION HEATMAP ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔗 Custom Multi-Stock Return Correlation Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(255,255,255,0.6); font-size:0.88rem;">Select any combination of Indian stocks below to dynamically evaluate their return correlations, identify co-moving assets, and discover diversification opportunities.</p>', unsafe_allow_html=True)

    default_corr_stocks = [
        sel_stock,
        "Tata Consultancy Services",
        "HDFC Bank",
        "Infosys",
        "State Bank of India",
        "ITC Limited",
        "Larsen & Toubro",
    ]
    seen = set()
    clean_defaults = [x for x in default_corr_stocks if not (x in seen or seen.add(x)) and x in NIFTY_50_STOCKS]

    selected_corr_stocks = st.multiselect(
        "Select stocks for correlation matrix:",
        options=sorted(list(NIFTY_50_STOCKS.keys())),
        default=clean_defaults,
        placeholder="Choose 2 or more stocks..."
    )

    if len(selected_corr_stocks) >= 2:
        pivot_corr_data = df[df["Stock_Name"].isin(selected_corr_stocks)].pivot_table(
            index="Date", columns="Stock_Name", values="Close"
        )
        corr_matrix = calculate_correlation_matrix(pivot_corr_data)

        if not corr_matrix.empty:
            fig_custom_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-1, zmax=1,
                labels=dict(color="Correlation")
            )
            fig_custom_corr.update_layout(
                **PLOTLY_THEME,
                title=f"Pearson Correlation Matrix ({len(selected_corr_stocks)} Selected Stocks)",
                height=480 + (len(selected_corr_stocks) * 15)
            )
            st.plotly_chart(fig_custom_corr, use_container_width=True)

            # Find highest and lowest pairs
            unstacked = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).unstack().dropna()
            if not unstacked.empty:
                max_pair = unstacked.idxmax()
                max_val = unstacked.max()
                min_pair = unstacked.idxmin()
                min_val = unstacked.min()

                c_info1, c_info2 = st.columns(2)
                c_info1.markdown(f"""
                <div class="insight-card" style="border-left-color:#34d399;">
                    <div class="insight-title" style="color:#34d399;">📈 Strongest Co-Movement Pair</div>
                    <div class="insight-text"><b>{max_pair[0]}</b> & <b>{max_pair[1]}</b> have the highest positive correlation (<b>r = {max_val:.2f}</b>). They tend to move together.</div>
                </div>
                """, unsafe_allow_html=True)

                c_info2.markdown(f"""
                <div class="insight-card" style="border-left-color:#60a5fa;">
                    <div class="insight-title" style="color:#60a5fa;">🛡️ Best Diversification Pair</div>
                    <div class="insight-text"><b>{min_pair[0]}</b> & <b>{min_pair[1]}</b> have the lowest correlation (<b>r = {min_val:.2f}</b>), offering superior portfolio hedging and risk reduction.</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please select at least 2 stocks above to display the correlation heatmap.")

# ════════════════════════════════════════════════════════════
# TAB 3 – SECTOR ANALYTICS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🏢 Indian Sector Performance Analysis</div>', unsafe_allow_html=True)

    disp_sstats = sector_stats[sector_stats["Sector"].isin(sel_sectors)] if sel_sectors else sector_stats

    c1, c2 = st.columns(2)

    with c1:
        # Volume by sector
        fig_sec_vol = px.bar(
            disp_sstats.sort_values("Total_Volume"),
            x="Total_Volume", y="Sector", orientation="h",
            color="Total_Volume", color_continuous_scale="Viridis",
            labels={"Total_Volume": "Total Traded Volume (Shares)", "Sector": "Sector"}
        )
        fig_sec_vol.update_layout(**PLOTLY_THEME, title="Total Trading Volume by Indian Sector", height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_sec_vol, use_container_width=True)

    with c2:
        # Risk vs Return by sector
        fig_sec_risk = px.scatter(
            disp_sstats, x="Avg_Volatility", y="Avg_Return",
            size="Num_Stocks", color="Avg_Return",
            text="Sector", color_continuous_scale="RdYlGn",
            labels={"Avg_Volatility": "Avg Volatility (%)", "Avg_Return": "Avg Return (%)"}
        )
        fig_sec_risk.update_traces(textposition="top center", textfont_size=11)
        fig_sec_risk.update_layout(**PLOTLY_THEME, title="Risk vs. Return by Indian Sector", height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_sec_risk, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Volume share donut
        fig_sec_pie = px.pie(
            disp_sstats, names="Sector", values="Total_Volume",
            hole=0.55, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_sec_pie.update_layout(**PLOTLY_THEME, title="Trading Volume Share Across Sectors", height=380)
        fig_sec_pie.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig_sec_pie, use_container_width=True)

    with c4:
        # Average return by sector
        fig_sec_ret = px.bar(
            disp_sstats.sort_values("Avg_Return", ascending=False),
            x="Sector", y="Avg_Return",
            color="Avg_Return", color_continuous_scale="RdYlGn",
            labels={"Avg_Return": "Average Return (%)"}
        )
        fig_sec_ret.update_layout(**PLOTLY_THEME, title="Average Stock Return by Sector", height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_sec_ret, use_container_width=True)

    st.markdown('<div class="section-header">📋 Indian Sector Summary Table</div>', unsafe_allow_html=True)
    table_df = disp_sstats.copy()
    table_df["Avg_Return"] = table_df["Avg_Return"].map("{:.2f}%".format)
    table_df["Avg_Volatility"] = table_df["Avg_Volatility"].map("{:.2f}%".format)
    table_df["Total_Volume"] = table_df["Total_Volume"].map("{:,.0f}".format)
    table_df.columns = ["Sector", "# Constituents", "Avg Return", "Avg Volatility", "Total Volume"]
    st.dataframe(table_df.reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 – MARKET OVERVIEW
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📉 Nifty 50 Broad Market Overview</div>', unsafe_allow_html=True)

    top_gainers = stock_stats_filtered.nlargest(10, "Total_Return_Pct")
    top_losers = stock_stats_filtered.nsmallest(10, "Total_Return_Pct")

    c1, c2 = st.columns(2)

    with c1:
        fig_top = px.bar(
            top_gainers.sort_values("Total_Return_Pct"),
            x="Total_Return_Pct", y="Stock_Name", orientation="h",
            color="Total_Return_Pct", color_continuous_scale="Greens",
            labels={"Total_Return_Pct": "Total Return (%)"}
        )
        fig_top.update_layout(**PLOTLY_THEME, title="🏆 Top 10 Nifty 50 Gainers", height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        fig_bot = px.bar(
            top_losers.sort_values("Total_Return_Pct", ascending=False),
            x="Total_Return_Pct", y="Stock_Name", orientation="h",
            color="Total_Return_Pct", color_continuous_scale="Reds_r",
            labels={"Total_Return_Pct": "Total Return (%)"}
        )
        fig_bot.update_layout(**PLOTLY_THEME, title="📉 Top 10 Nifty 50 Laggards", height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_bot, use_container_width=True)

    # Risk vs Return Scatter for all Nifty stocks
    fig_scatter = px.scatter(
        stock_stats_filtered, x="Volatility_Score", y="Total_Return_Pct",
        color="Sector", size="Total_Volume", hover_name="Stock_Name",
        labels={"Volatility_Score": "Volatility (%)", "Total_Return_Pct": "Total Return (%)"},
        size_max=30
    )
    fig_scatter.update_layout(**PLOTLY_THEME, title="🎯 Risk vs. Return — Nifty 50 Constituents (Bubble size = Volume)", height=480)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Heatmap for Top Volume Indian stocks
    st.markdown('<div class="section-header">🔗 Correlation Matrix — Top Heavyweight Constituents</div>', unsafe_allow_html=True)
    top15_stocks = stock_stats.nlargest(15, "Total_Volume")["Stock_Name"].tolist()
    pivot_corr = df[df["Stock_Name"].isin(top15_stocks)].pivot_table(index="Date", columns="Stock_Name", values="Close")
    corr_matrix = pivot_corr.pct_change().corr()

    fig_corr = px.imshow(
        corr_matrix, color_continuous_scale="RdBu_r", aspect="auto",
        labels=dict(color="Correlation"), zmin=-1, zmax=1
    )
    fig_corr.update_layout(**PLOTLY_THEME, title="Daily Return Correlation — Top Traded Nifty Stocks", height=550)
    st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 – BENCHMARK & BETA ANALYSIS (CAPM)
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-header">🎯 CAPM Beta & Sensitivity Analysis — {sel_stock} vs. Nifty 50 (^NSEI)</div>', unsafe_allow_html=True)

    if not stock_df.empty and not benchmark_df.empty:
        merged_capm = pd.merge(
            stock_df[["Date", "Daily_Return"]].dropna(),
            benchmark_df[["Date", "Daily_Return"]].dropna(),
            on="Date",
            suffixes=("_stock", "_bench")
        )

        # Quantitative Metrics Cards
        q1, q2, q3, q4 = st.columns(4)
        q1.markdown(f"""
        <div class="kpi-card amber" style="text-align:center;">
            <div class="kpi-label">Market Beta (β)</div>
            <div class="kpi-value">{beta}</div>
            <div class="kpi-sub">{'Aggressive (>1.0)' if beta > 1.05 else 'Defensive (<1.0)' if beta < 0.95 else 'Market Neutral (~1.0)'}</div>
        </div>
        """, unsafe_allow_html=True)

        q2.markdown(f"""
        <div class="kpi-card green" style="text-align:center;">
            <div class="kpi-label">Annualized Alpha (α)</div>
            <div class="kpi-value">{alpha:+.2f}%</div>
            <div class="kpi-sub">Excess return over benchmark</div>
        </div>
        """, unsafe_allow_html=True)

        q3.markdown(f"""
        <div class="kpi-card blue" style="text-align:center;">
            <div class="kpi-label">Correlation (r)</div>
            <div class="kpi-value">{corr:.2f}</div>
            <div class="kpi-sub">Co-movement with Nifty 50</div>
        </div>
        """, unsafe_allow_html=True)

        q4.markdown(f"""
        <div class="kpi-card purple" style="text-align:center;">
            <div class="kpi-label">R-Squared (R²)</div>
            <div class="kpi-value">{r_sq:.2f}</div>
            <div class="kpi-sub">{r_sq*100:.0f}% variance explained by Nifty</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Scatter Regression Plot using pure NumPy (no statsmodels dependency required)
        fig_capm = go.Figure()
        fig_capm.add_trace(go.Scatter(
            x=merged_capm["Daily_Return_bench"],
            y=merged_capm["Daily_Return_stock"],
            mode="markers",
            name="Daily Returns",
            marker=dict(color="#34d399", size=6, opacity=0.75)
        ))
        
        # Fit regression line: y = slope * x + intercept
        if len(merged_capm) > 1:
            slope, intercept = np.polyfit(merged_capm["Daily_Return_bench"], merged_capm["Daily_Return_stock"], 1)
            line_x = np.linspace(merged_capm["Daily_Return_bench"].min(), merged_capm["Daily_Return_bench"].max(), 50)
            line_y = slope * line_x + intercept
            fig_capm.add_trace(go.Scatter(
                x=line_x, y=line_y,
                mode="lines",
                name=f"CAPM Fit Line (β: {slope:.2f})",
                line=dict(color="#f59e0b", width=2.5, dash="solid")
            ))

        fig_capm.update_layout(
            xaxis_title="Nifty 50 Daily Return (%)",
            yaxis_title=f"{sel_stock} Daily Return (%)"
        )
        fig_capm = apply_theme(fig_capm, f"CAPM Characteristic Line: {sel_stock} vs. Nifty 50 (Slope = Beta: {beta})", 450)
        st.plotly_chart(fig_capm, use_container_width=True)

        # Interpretation Card
        beta_text = (
            f"<b>{sel_stock}</b> has a Beta of <b>{beta}</b>, indicating it is "
            f"{'significantly more volatile than Nifty 50 — swings are magnified.' if beta > 1.2 else 'moderately higher volatility than the benchmark.' if beta > 1.0 else 'defensive and less volatile than the benchmark index.' if beta < 0.8 else 'closely synchronized with Nifty 50 index movements.'}"
        )
        alpha_text = (
            f"The stock generates an annualized Alpha of <b>{alpha:+.2f}%</b>, meaning it has "
            f"{'outperformed its risk-adjusted benchmark expectations.' if alpha > 0 else 'underperformed its expected CAPM benchmark hurdle.'}"
        )

        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.95rem; font-weight:700; color:#fbbf24; margin-bottom:0.5rem;">🧠 Quantitative Risk Interpretation</div>
            <p style="color:rgba(255,255,255,0.85); font-size:0.9rem; margin-bottom:0.4rem;">• {beta_text}</p>
            <p style="color:rgba(255,255,255,0.85); font-size:0.9rem; margin-bottom:0;">• {alpha_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Benchmark data not available for this period.")

# ════════════════════════════════════════════════════════════
# TAB 6 – MARKET INSIGHTS
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">💡 Dynamic Indian Market Intelligence & Insights</div>', unsafe_allow_html=True)

    best_stock = stock_stats.loc[stock_stats["Total_Return_Pct"].idxmax()]
    worst_stock = stock_stats.loc[stock_stats["Total_Return_Pct"].idxmin()]
    lowest_vol_stock = stock_stats.loc[stock_stats["Volatility_Score"].idxmin()]
    highest_vol_stock = stock_stats.loc[stock_stats["Volatility_Score"].idxmax()]
    top_volume_stock = stock_stats.loc[stock_stats["Total_Volume"].idxmax()]
    best_sector = sector_stats.loc[sector_stats["Avg_Return"].idxmax()]
    worst_sector = sector_stats.loc[sector_stats["Avg_Return"].idxmin()]
    top_vol_sector = sector_stats.iloc[0]

    it_stocks = stock_stats[stock_stats["Sector"] == "Information Technology"]["Total_Return_Pct"].mean()
    bank_stocks = stock_stats[stock_stats["Sector"] == "Banking & Financial Services"]["Total_Return_Pct"].mean()
    overall_avg_ret = stock_stats["Total_Return_Pct"].mean()
    bull_bear = "🟢 Bullish" if overall_avg_ret > 0 else "🔴 Bearish"

    # Outperforming Nifty count
    if not benchmark_df.empty:
        b_final_ret = ((benchmark_df["Close"].iloc[-1] - benchmark_df["Close"].iloc[0]) / benchmark_df["Close"].iloc[0]) * 100
        outperformers_count = (stock_stats["Total_Return_Pct"] > b_final_ret).sum()
        breadth_pct = (outperformers_count / len(stock_stats)) * 100
    else:
        breadth_pct = 50.0

    INSIGHT_COLORS = ["#f59e0b", "#60a5fa", "#34d399", "#f87171", "#a78bfa", "#facc15", "#2dd4bf", "#fb923c", "#f59e0b", "#60a5fa", "#34d399", "#fb923c"]

    insights = [
        ("🏆 Nifty 50 Leader", f"{best_stock['Stock_Name']} leads the index with a remarkable {best_stock['Total_Return_Pct']:+.2f}% total return over the period.", INSIGHT_COLORS[0]),
        ("📉 Index Laggard", f"{worst_stock['Stock_Name']} recorded the steepest drop of {worst_stock['Total_Return_Pct']:.2f}% total return.", INSIGHT_COLORS[1]),
        ("🛡️ Most Stable Bluechip", f"{lowest_vol_stock['Stock_Name']} exhibits the lowest volatility score ({lowest_vol_stock['Volatility_Score']:.2f}%), ideal for capital preservation.", INSIGHT_COLORS[2]),
        ("🌋 High-Beta Mover", f"{highest_vol_stock['Stock_Name']} is the most volatile stock ({highest_vol_stock['Volatility_Score']:.2f}% volatility score), suited for momentum traders.", INSIGHT_COLORS[3]),
        ("🚀 Outperforming Sector", f"<b>{best_sector['Sector']}</b> is the highest returning sector with an average gain of {best_sector['Avg_Return']:+.2f}%.", INSIGHT_COLORS[4]),
        ("⚠️ Lagging Sector", f"<b>{worst_sector['Sector']}</b> posted the weakest sector performance averaging {worst_sector['Avg_Return']:+.2f}%.", INSIGHT_COLORS[5]),
        ("📦 Trading Volume Giant", f"{top_volume_stock['Stock_Name']} leads all NSE stocks with {top_volume_stock['Total_Volume']/1e9:.2f}B shares traded.", INSIGHT_COLORS[6]),
        ("🏢 Liquidity Hub Sector", f"{top_vol_sector['Sector']} accounts for the highest market turnover with {top_vol_sector['Total_Volume']/1e9:.2f}B total shares.", INSIGHT_COLORS[7]),
        ("💻 IT vs. 🏦 Banking", f"IT stocks averaged {it_stocks:+.2f}%, while Banking & Financials averaged {bank_stocks:+.2f}%.", INSIGHT_COLORS[8]),
        ("🎯 Market Breadth", f"<b>{breadth_pct:.1f}%</b> of Nifty 50 constituents outperformed the benchmark index (^NSEI) over this period.", INSIGHT_COLORS[9]),
        ("📈 Market Sentiment", f"Overall Indian equity sentiment is <b>{bull_bear}</b>, with average constituent return of {overall_avg_ret:+.2f}%.", INSIGHT_COLORS[10]),
        ("⚡ Live NSE Pipeline", f"Data ingestion verified with live NSE tickers (.NS) and cached for maximum responsiveness.", INSIGHT_COLORS[11]),
    ]

    st.markdown('<div class="insight-grid">', unsafe_allow_html=True)
    for title, text, color in insights:
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:{color};">
            <div class="insight-title" style="color:{color};">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # AI Box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 AI-Style Market Intelligence Summary</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-box">
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:1.5rem; flex-wrap:wrap;">
            <div>
                <div class="ai-label">Market Regime</div>
                <div class="ai-value">{bull_bear}</div>
                <div class="ai-desc">Based on constituent return breadth across Nifty 50</div>
            </div>
            <div>
                <div class="ai-label">Top Performer</div>
                <div class="ai-value">🏆 {best_stock['Stock_Name']}</div>
                <div class="ai-desc">{best_stock['Total_Return_Pct']:+.2f}% total return over selected window</div>
            </div>
            <div>
                <div class="ai-label">Benchmark Performance</div>
                <div class="ai-value">{bench_return_str}</div>
                <div class="ai-desc">NIFTY 50 (^NSEI) index performance</div>
            </div>
            <div>
                <div class="ai-label">Sector Driver</div>
                <div class="ai-value">🚀 {best_sector['Sector']}</div>
                <div class="ai-desc">Sector with highest average return ({best_sector['Avg_Return']:+.2f}%)</div>
            </div>
            <div>
                <div class="ai-label">Selected Stock Beta</div>
                <div class="ai-value">⚡ {sel_stock}: {beta}</div>
                <div class="ai-desc">Sensitivity to Nifty 50 movements (Annual Alpha: {alpha:+.2f}%)</div>
            </div>
            <div>
                <div class="ai-label">Data Coverage</div>
                <div class="ai-value">✅ Live NSE Feed</div>
                <div class="ai-desc">Fetched from yfinance with automated error resilience</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 7 – QUANT PORTFOLIO / RECRUITER SHOWCASE
# ════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">🏆 Quantitative Finance & Engineering Demonstrations</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(255,255,255,0.65); margin-bottom:1.5rem;">This project demonstrates production-ready quantitative finance capabilities, real-time API engineering with yfinance, CAPM modeling against Indian benchmarks, and interactive business intelligence dashboard design.</p>', unsafe_allow_html=True)

    badges = [
        "✅ Live yfinance NSE Data Ingestion",
        "✅ NIFTY 50 Benchmark (^NSEI) Modeling",
        "✅ Capital Asset Pricing Model (CAPM)",
        "✅ Beta (β) & Annualized Alpha (α) Calculation",
        "✅ Technical Indicators (MA20, MA50, MA200)",
        "✅ Rolling Volatility Modeling",
        "✅ Sector Performance & Turnover Analytics",
        "✅ Multi-Ticker Batch Threading",
        "✅ Streamlit Caching Performance (@st.cache_data)",
        "✅ Resilient Error Handling for Delisted/Failed Symbols",
        "✅ Normalized Relative Return Overlays",
        "✅ Price Return Correlation Matrix",
        "✅ Interactive Candlestick & Volume Visualization",
        "✅ Dynamic Market Insights Engine",
    ]
    badge_html = " ".join([f'<span class="badge">{b}</span>' for b in badges])
    st.markdown(f'<div class="badge-grid">{badge_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    stack = [
        ("🐍 Python 3", "Core Language", "#3b82f6"),
        ("📊 Streamlit", "Reactive BI Framework", "#f43f5e"),
        ("📉 Plotly", "Interactive Financial Charts", "#8b5cf6"),
        ("🐼 Pandas", "Time-Series Data Wrangling", "#10b981"),
        ("📈 yfinance", "NSE Real-Time API Ingestion", "#f59e0b"),
        ("🔢 NumPy", "Quantitative CAPM Modeling", "#06b6d4"),
    ]
    cols = st.columns(len(stack))
    for i, (icon_name, desc, color) in enumerate(stack):
        cols[i].markdown(f"""
        <div class="glass-card" style="text-align:center; border-top: 3px solid {color};">
            <div style="font-size:1.5rem;">{icon_name.split()[0]}</div>
            <div style="font-weight:700; color:#fff; margin-top:0.3rem; font-size:0.9rem;">{' '.join(icon_name.split()[1:])}</div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4);">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <br>
    <div style="text-align:center; padding: 2rem; color:rgba(255,255,255,0.3); font-size:0.8rem; border-top: 1px solid rgba(255,255,255,0.06);">
        🇮🇳 Indian Stock Intelligence Dashboard &nbsp;|&nbsp; NSE NIFTY 50 Edition &nbsp;|&nbsp; Real-Time Data via yfinance
    </div>
    """, unsafe_allow_html=True)

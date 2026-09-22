import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import requests

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nifty 500 Momentum Strategy",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Nifty 500 Sector-Based Momentum Analyzer")
st.markdown("Download universe, calculate 3M/6M/12M returns, and rank stocks relative to their sector peers.")


# ---------------------------------------------------------
# Step 1: Universe Downloader (Cached)
# ---------------------------------------------------------
@st.cache_data(ttl=86400)  # Cache for 24 hours
def download_nifty500_raw():
    """Attempts to fetch the list directly from the official NiftyIndices portal."""
    url = "https://niftyindices.com"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        # Intercept HTML proxy blocks or firewall pages
        if "html" in response.text.lower() or response.status_code != 200:
            return None
        return pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    except Exception:
        return None


def parse_nifty_dataframe(df_symbols):
    """Sanitizes columns dynamically regardless of case spacing formatting."""
    df_symbols.columns = df_symbols.columns.str.strip()

    # Flexible column matching
    symbol_col = [c for c in df_symbols.columns if 'symbol' in c.lower()]
    name_col = [c for c in df_symbols.columns if 'company name' in c.lower()]
    sector_col = [c for c in df_symbols.columns if 'industry' in c.lower() or 'sector' in c.lower()]

    if not symbol_col:
        st.error("Failed to parse file: Could not locate a stock ticker 'Symbol' column.")
        return pd.DataFrame()

    target_symbol = symbol_col[0]
    target_name = name_col[0] if name_col else df_symbols.columns[0]
    target_sector = sector_col[0] if sector_col else None

    df_symbols['Ticker'] = df_symbols[target_symbol].astype(str).str.strip() + '.NS'

    if target_sector:
        universe_df = df_symbols[['Ticker', target_name, target_sector]].rename(
            columns={target_name: 'Company Name', target_sector: 'Sector'}
        )
    else:
        df_symbols['Sector'] = 'General Market'
        universe_df = df_symbols[['Ticker', target_name, 'Sector']].rename(columns={target_name: 'Company Name'})

    return universe_df


# ---------------------------------------------------------
# Step 2: Momentum Return Calculator (Vectorised Batch Download)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache calculations for 1 hour
def calculate_momentum(tickers):
    if not tickers:
        return pd.DataFrame()

    try:
        # Bulk download 1 year of historical daily Close data
        data = yf.download(tickers, period="1y", interval="1d", group_by='column', auto_adjust=True)
        close_prices = data['Close'] if 'Close' in data else data
    except Exception as e:
        st.error(f"Error bulk-downloading data from Yahoo Finance: {e}")
        return pd.DataFrame()

    close_prices = close_prices.ffill()
    momentum_metrics = []

    # Trading day intervals mapping (3M ~ 60d, 6M ~ 125d, 1Y ~ 250d)
    for ticker in tickers:
        if ticker in close_prices.columns:
            series = close_prices[ticker].dropna()
            if len(series) > 10:
                current_price = series.iloc[-1]

                p_3m = series.iloc[-min(60, len(series))]
                p_6m = series.iloc[-min(125, len(series))]
                p_12m = series.iloc[-min(250, len(series))]

                ret_3m = (current_price / p_3m) - 1 if p_3m > 0 else np.nan
                ret_6m = (current_price / p_6m) - 1 if p_6m > 0 else np.nan
                ret_12m = (current_price / p_12m) - 1 if p_12m > 0 else np.nan

                avg_momentum = np.nanmean([ret_3m, ret_6m, ret_12m])

                momentum_metrics.append({
                    'Ticker': ticker,
                    'Current Price': round(current_price, 2),
                    '3M Return (%)': round(ret_3m * 100, 2),
                    '6M Return (%)': round(ret_6m * 100, 2),
                    '12M Return (%)': round(ret_12m * 100, 2),
                    'Composite Score': round(avg_momentum * 100, 2)
                })

    return pd.DataFrame(momentum_metrics)


# ---------------------------------------------------------
# Sidebar Setup & Data Ingestion Engine
# ---------------------------------------------------------
st.sidebar.header("Data Ingestion Method")
uploaded_file = st.sidebar.file_uploader(
    "Option B: Manual CSV Backup Upload (If live download fails)",
    type=["csv"],
    help="Download the index list from NiftyIndices or NSE website, then drop it here."
)

raw_df = None
universe_df = pd.DataFrame()

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("Loaded backup dataset from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
else:
    with st.spinner("Fetching live Nifty 500 constituency list..."):
        raw_df = download_nifty500_raw()
    if raw_df is not None:
        st.sidebar.success("Live list loaded from NiftyIndices server.")
    else:
        st.sidebar.warning(
            "Live download blocked by firewall. Please upload the CSV manually or testing with Fallback.")

# Process Data or load mock fallback array if everything is empty
if raw_df is not None:
    universe_df = parse_nifty_dataframe(raw_df)
else:
    fallback_data = {
        'Ticker': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'LT.NS', 'BHARTIARTL.NS',
                   'ITC.NS', 'TATAMOTORS.NS', 'SUNPHARMA.NS'],
        'Company Name': ['Reliance Industries', 'TCS', 'HDFC Bank', 'Infosys', 'ICICI Bank', 'L&T', 'Bharti Airtel',
                         'ITC', 'Tata Motors', 'Sun Pharma'],
        'Sector': ['Energy', 'IT', 'Financial Services', 'IT', 'Financial Services', 'Construction',
                   'Telecommunication', 'FMCG', 'Automobile', 'Healthcare']
    }
    universe_df = pd.DataFrame(fallback_data)
    st.sidebar.info("Running on standard blueprint fallback ticker index.")

# ---------------------------------------------------------
# Processing Pipeline & Presentation UI Layout
# ---------------------------------------------------------
if not universe_df.empty:
    ticker_list = universe_df['Ticker'].tolist()

    st.subheader("Processing Analytics Engine")
    with st.spinner("Downloading historical data & executing cross-sectional ranking scans..."):
        momentum_df = calculate_momentum(ticker_list)

    if not momentum_df.empty:
        # Merge stats and compute structural group ranks
        final_analysis = pd.merge(universe_df, momentum_df, on='Ticker', how='inner')
        final_analysis['Sector Rank'] = final_analysis.groupby('Sector')['Composite Score'].rank(ascending=False,
                                                                                                 method='min').astype(
            int)
        final_analysis = final_analysis.sort_values(by='Composite Score', ascending=False).reset_index(drop=True)

        # Display Controls
        st.sidebar.header("Filter & Strategy Configuration")
        selected_sector = st.sidebar.selectbox(
            "Filter View by Sector Peer Group:",
            ["All Sectors"] + sorted(final_analysis['Sector'].unique().tolist())
        )
        top_n = st.sidebar.slider("Limit Rows Displayed (Top Performers):", min_value=10, max_value=len(final_analysis),
                                  value=50)

        # Apply Filters Dynamically
        display_df = final_analysis.copy()
        if selected_sector != "All Sectors":
            display_df = display_df[display_df['Sector'] == selected_sector]
            display_df['Sector Rank'] = display_df['Composite Score'].rank(ascending=False, method='min').astype(int)
            display_df = display_df.sort_values(by='Sector Rank', ascending=True)

        display_df = display_df.head(top_n)

        # Metric Layout Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks Tracked", len(final_analysis))
        with col2:
            st.metric("Highest Composite Momentum Score", f"{final_analysis['Composite Score'].max()}%")
        with col3:
            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Filtered Table to CSV",
                data=csv_buffer.getvalue(),
                file_name="nifty_momentum_rankings.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Main Table Presentation
        st.dataframe(
            display_df[
                ['Sector Rank', 'Ticker', 'Company Name', 'Sector', 'Current Price', '3M Return (%)', '6M Return (%)',
                 '12M Return (%)', 'Composite Score']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Analysis calculation failed. No assets returned tracking data from yfinance.")

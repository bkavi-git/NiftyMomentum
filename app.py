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
@st.cache_data(ttl=86400) # Cache for 24 hours
def fetch_nifty500_universe():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        df_symbols = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        df_symbols['Ticker'] = df_symbols['Symbol'].astype(str) + '.NS'
        sector_col = [col for col in df_symbols.columns if 'Industry' in col or 'Sector' in col][0]
        universe_df = df_symbols[['Ticker', 'Company Name', sector_col]].rename(columns={sector_col: 'Sector'})
        return universe_df
    except Exception as e:
        st.error(f"Failed to fetch Nifty 500 list directly from NSE: {e}")
        # Fallback list for testing
        data = {
            'Ticker': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'LT.NS', 'BHARTIARTL.NS', 'ITC.NS', 'TATAMOTORS.NS', 'SUNPHARMA.NS'],
            'Company Name': ['Reliance Ind', 'TCS', 'HDFC Bank', 'Infosys', 'ICICI Bank', 'L&T', 'Airtel', 'ITC', 'Tata Motors', 'Sun Pharma'],
            'Sector': ['Energy', 'IT', 'Financials', 'IT', 'Financials', 'Construction', 'Telecom', 'FMCG', 'Automobile', 'Healthcare']
        }
        return pd.DataFrame(data)

# ---------------------------------------------------------
# Step 2: Data Download & Momentum Calculation (Cached)
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # Cache price data for 1 hour
def compute_sector_momentum():
    universe = fetch_nifty500_universe()
    tickers = universe['Ticker'].tolist()

    # Download 13 months of daily close prices
    price_data = yf.download(
        tickers=tickers,
        period="1y3m",
        interval="1d",
        auto_adjust=True,
        progress=False
    )['Close']

    # Clean missing data
    price_data = price_data.dropna(thresh=int(len(price_data) * 0.8), axis=1)

    LOOKBACK_3M = 63
    LOOKBACK_6M = 126
    LOOKBACK_12M = 252

    # Return calculations
    ret_3m = (price_data.iloc[-1] / price_data.iloc[-LOOKBACK_3M] - 1) * 100
    ret_6m = (price_data.iloc[-1] / price_data.iloc[-LOOKBACK_6M] - 1) * 100
    ret_12m = (price_data.iloc[-1] / price_data.iloc[-LOOKBACK_12M] - 1) * 100

    momentum_df = pd.DataFrame({
        '3M_Return_%': ret_3m,
        '6M_Return_%': ret_6m,
        '12M_Return_%': ret_12m
    }).reset_index().rename(columns={'index': 'Ticker'})

    result = pd.merge(universe, momentum_df, on='Ticker', how='inner')
    result = result.dropna(subset=['3M_Return_%', '6M_Return_%', '12M_Return_%']).copy()

    # Sector-wise Rankings
    for period in ['3M', '6M', '12M']:
        col_name = f'{period}_Return_%'
        rank_col = f'Sector_Rank_{period}'
        pct_rank_col = f'Sector_Percentile_{period}'
        
        # Absolute Sector Rank (1 = Top Return)
        result[rank_col] = result.groupby('Sector')[col_name].rank(ascending=False, method='dense').astype(int)
        
        # Relative Percentile within Sector (0 to 100, higher is better)
        result[pct_rank_col] = (result.groupby('Sector')[col_name].rank(pct=True) * 100).round(1)

    # Weighted Composite Score
    result['Composite_Score'] = (
        result['Sector_Percentile_3M'] * 0.20 +
        result['Sector_Percentile_6M'] * 0.30 +
        result['Sector_Percentile_12M'] * 0.50
    ).round(1)

    return result

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Controls & Filters")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Downloading price data and calculating sector momentum..."):
    df = compute_sector_momentum()

# Filter by Sector
sectors = ["All"] + sorted(df['Sector'].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector", sectors)

# Filter by Top N Ranked per Sector
top_n = st.sidebar.slider("Select Top N per Sector", min_value=1, max_value=20, value=5)

# Primary Sorting Factor
sort_factor = st.sidebar.selectbox(
    "Primary Sorting Rank", 
    ["Composite_Score", "Sector_Rank_6M", "Sector_Rank_12M", "Sector_Rank_3M"]
)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------

# Apply Filters
filtered_df = df.copy()
if selected_sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]

# Filter top N per sector
filtered_df = filtered_df[filtered_df['Sector_Rank_6M'] <= top_n]

if sort_factor == "Composite_Score":
    filtered_df = filtered_df.sort_values(by=sort_factor, ascending=False)
else:
    filtered_df = filtered_df.sort_values(by=['Sector', sort_factor], ascending=[True, True])

# Summary Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Stocks Analyzed", len(df))
col2.metric("Total Sectors", df['Sector'].nunique())
col3.metric("Filtered Output Count", len(filtered_df))
col4.metric("Top Return Stock (6M)", f"{df.loc[df['6M_Return_%'].idxmax()]['Company Name']} ({df['6M_Return_%'].max():.1f}%)")

st.divider()

# Tab Navigation
tab1, tab2 = st.tabs(["📊 Momentum Table", "📈 Sector Heatmap / Visuals"])

with tab1:
    st.subheader("Sector-Grouped Momentum Leaderboard")
    
    # Download Button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='nifty500_sector_momentum.csv',
        mime='text/csv'
    )
    
    # Interactive Data Table
    display_df = filtered_df[[
        'Company Name', 'Ticker', 'Sector', 
        '3M_Return_%', '6M_Return_%', '12M_Return_%',
        'Sector_Rank_3M', 'Sector_Rank_6M', 'Sector_Rank_12M', 'Composite_Score'
    ]].round(2)

    st.dataframe(
        display_df,
        column_config={
            "3M_Return_%": st.column_config.NumberColumn(format="%.2f%%"),
            "6M_Return_%": st.column_config.NumberColumn(format="%.2f%%"),
            "12M_Return_%": st.column_config.NumberColumn(format="%.2f%%"),
            "Composite_Score": st.column_config.ProgressColumn(
                "Composite Score", min_value=0, max_value=100, format="%.1f"
            )
        },
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("Top Performers by Sector (6M Return)")
    
    # Bar Chart for Top Picks
    import plotly.express as px
    
    fig = px.bar(
        filtered_df.head(30),
        x='Company Name',
        y='6M_Return_%',
        color='Sector',
        title="Top Stocks 6-Month Returns (%) by Sector",
        hover_data=['3M_Return_%', '12M_Return_%', 'Sector_Rank_6M']
    )
    st.plotly_chart(fig, use_container_width=True)

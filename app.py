import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Minimalist black theme styling
st.markdown("""
    <style>
    body {
        background-color: #1c1c1c;
        color: #e0e0e0;
    }
    .css-1v3fvcr {
        background-color: #1c1c1c !important;
    }
    .stButton>button {
        background-color: #00BFFF !important;
        color: white !important;
    }
    .css-2trqyj {
        color: white !important;
    }
    h1, h2, h3, h4, h5 {
        color: #00BFFF;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions for technical indicators
def calculate_sma(data, period=50):
    return data['Close'].rolling(window=period).mean()

def calculate_rsi(data, period=20):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    sma = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return upper_band, lower_band

# Caching stock data fetching for 5 years
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)

# Caching company information fetching
@st.cache_data
def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    return {
        "Sector": stock.info.get("sector", "N/A"),
        "Country": stock.info.get("country", "N/A"),
        "PE Ratio": stock.info.get("trailingPE", "N/A"),
        "PB Ratio": stock.info.get("priceToBook", "N/A"),
        "EPS": stock.info.get("trailingEps", "N/A"),
    }

# Caching model training
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=10)  # Reduced number of estimators for faster training
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    # Sidebar for ticker input and additional options
    ticker = st.sidebar.text_input("Stock symbol:", "AAPL")

    # Moving Average settings
    short_ma_days = st.sidebar.slider("Short-term moving average days:", 10, 100, 10)
    long_ma_days = st.sidebar.slider("Long-term moving average days:", 50, 200, 50)
    
    # Date range for fetching data
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    # Load stock data and company info
    stock_data = load_stock_data(ticker, start_date, end_date)
    company_info = get_company_info(ticker)
    
    # 1. Company Information
    st.header("Company Information")
    st.write("### Basic Info")
    st.write(f"**Sector:** {company_info['Sector']}")
    st.write(f"**Country:** {company_info['Country']}")
    
    # 2. Fundamental Analysis
    st.header("Fundamental Analysis")
    st.write(f"**P/E Ratio:** {company_info['PE Ratio']}")
    st.write(f"**P/B Ratio:** {company_info['PB Ratio']}")
    st.write(f"**EPS:** {company_info['EPS']}")
    
    # Recommendations based on P/E ratio
    st.write("### Recommendation")
    if company_info['PE Ratio'] != "N/A" and company_info['PE Ratio'] < 15:
        recommendation = "Buy"
        description = "Undervalued"
    elif company_info['PE Ratio'] != "N/A" and company_info['PE Ratio'] > 25:
        recommendation = "Sell"
        description = "Overvalued"
    else:
        recommendation = "Hold"
        description = "Fairly Valued"
    
    st.write(f"**Recommendation:** {recommendation}")
    st.write(f"**Description:** {description}")
    
    # 3. Technical Analysis
    st.header("Technical Analysis")
    
    # Calculate custom SMA, RSI, MACD, Bollinger Bands based on user input
    stock_data['SMA_Short'] = calculate_sma(stock_data, short_ma_days)
    stock_data['SMA_Long'] = calculate_sma(stock_data, long_ma_days)
    stock_data['RSI'] = calculate_rsi(stock_data, 20)
    stock_data['MACD'] = calculate_macd(stock_data)
    stock_data['Upper_Band'], stock_data['Lower_Band'] = calculate_bollinger_bands(stock_data)

    # Crossover: Check for SMA crossover
    stock_data['Crossover'] = np.where(stock_data['SMA_Short'] > stock_data['SMA_Long'], 1, 0)
    stock_data['Crossover_Signal'] = stock_data['Crossover'].diff()

    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Close Price"))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_Short'], mode='lines', name=f"SMA {short_ma_days}"))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_Long'], mode='lines', name=f"SMA {long_ma_days}"))
    fig.update_layout(title=f"Close Price with SMA {short_ma_days} and SMA {long_ma_days}", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    st.write("### RSI (20)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name="RSI"))
    fig_rsi.update_layout(title="RSI (20)", template="plotly_dark")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    st.write("### MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name="MACD"))
    fig_macd.update_layout(title="MACD", template="plotly_dark")
    st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands
    st.write("### Bollinger Bands")
    fig_bollinger = go.Figure()
    fig_bollinger.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Close Price"))
    fig_bollinger.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'], mode='lines', name="Upper Band", line=dict(dash='dash')))
    fig_bollinger.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'], mode='lines', name="Lower Band", line=dict(dash='dash')))
    fig_bollinger.update_layout(title="Bollinger Bands", template="plotly_dark")
    st.plotly_chart(fig_bollinger, use_container_width=True)

    # 4. Prediction Model
    st.header("Prediction Model")
    stock_data['Target'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
    
    # Use relevant features for prediction and drop rows with missing values
    feature_columns = ['SMA_Short', 'SMA_Long', 'RSI', 'MACD']
    stock_data = stock_data.dropna(subset=feature_columns + ['Target'])
    
    features = stock_data[feature_columns]
    target = stock_data['Target']
    
    # Check if there is enough data for train_test_split
    if len(features) < 5:  # Adjust threshold based on

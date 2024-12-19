# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import date
import plotly.graph_objects as go

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to fetch company financials
def fetch_company_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        return financials
    except Exception as e:
        st.error(f"Error fetching financials: {e}")
        return None

# Function to fetch recent news
def fetch_recent_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return None

# Function to preprocess the data
def preprocess_data(data):
    # Drop any rows with missing values
    data = data.dropna()
    
    # Create features and target
    X = np.array(range(len(data))).reshape(-1, 1)  # Use index as feature
    y = data['Close'].values
    
    return X, y

# Function to train the linear regression model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to make predictions
def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

# Streamlit app
st.title("Indian Stock Market Prediction App")
st.write("This app predicts stock prices using a simple Linear Regression model.")

# User inputs
stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")
start_date = st.date_input("Start Date", value=date(2015, 1, 1))
end_date = st.date_input("End Date", value=date(2024, 12, 13))

# Fetch and display stock data
if st.button("Fetch and Predict"):
    st.write(f"Fetching data for {stock_ticker}...")
    stock_data = fetch_stock_data(stock_ticker, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        st.write("Stock Data:")
        st.dataframe(stock_data.tail())
        
        # Plot stock prices
        st.write("Closing Price Trend:")
        st.line_chart(stock_data['Close'])
        
        # Moving Averages
        st.write("Moving Averages (20 and 50 days):")
        stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Close'], label="Close Price", color="blue")
        plt.plot(stock_data['MA20'], label="20-Day MA", color="orange")
        plt.plot(stock_data['MA50'], label="50-Day MA", color="green")
        plt.title(f"{stock_ticker} Price and Moving Averages")
        plt.legend()
        st.pyplot(plt)

        # Candlestick chart
        st.write("Candlestick Chart:")
        fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                               open=stock_data['Open'],
                                               high=stock_data['High'],
                                               low=stock_data['Low'],
                                               close=stock_data['Close'])])
        fig.update_layout(title=f"{stock_ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Preprocess data
        st.write("Preprocessing data...")
        X, y = preprocess_data(stock_data)
        
        # Train the model
        st.write("Training Linear Regression model...")
        model = train_model(X, y)
        
        # Make predictions
        st.write("Making predictions...")
        future_dates = pd.date_range(start=end_date, periods=28, freq='D')
        future_X = np.array(range(len(stock_data), len(stock_data) + len(future_dates))).reshape(-1, 1)
        predictions = make_predictions(model, future_X)
        
        # Plot predictions vs actual
        st.write("Predictions vs Actual Prices:")
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data.index, stock_data['Close'], label="Actual Prices", color="blue")
        plt.plot(future_dates, predictions, label="Predicted Prices", color="red")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        st.pyplot(plt)

        # Fetch and display company financials
        st.write("Company Financials:")
        financials = fetch_company_financials(stock_ticker)
        if financials is not None:
            st.dataframe(financials)

        # Fetch and display recent news
        st.write("Recent News:")
        news = fetch_recent_news(stock_ticker)
        if news is not None:
            for article in news:
                st.write(f"**{article['title']}**")
                st.write(article['link'])
                st.write("---")
    else:
        st.error("No data found for the given stock ticker and date range.")

st.write("Abhinav Pyth")        
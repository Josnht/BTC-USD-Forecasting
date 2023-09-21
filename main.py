import streamlit as st 
from datetime import date

import numpy as np 
import keras
from keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt

import pandas as pd
from plotly import graph_objs as go
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Bitcoin Forecast App')


selected_ticker = ("BTC-USD")

model = load_model('./LSTM_model.h5')

scaler = joblib.load("./scaler.pkl")

@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_ticker)
data_load_state.text('Loading data... done!')

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="open_prices"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="close_prices"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
      
plot_raw_data()

# Input for the number of days to forecast
period = ('Choose the days','30 days')
n_days = st.selectbox('Days of prediction:', period)
forecast_horizon_days = 30 if n_days == '30 days' else None

def fetch_btc_data(num_days):
    try:
        # Fetch historical BTC data using yfinance
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=num_days)).strftime('%Y-%m-%d')
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        return btc_data
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return None

def preprocess_data(data):
    data['Date'] = data.index  # Use the date as a feature
    data['Date'] = data['Date'].astype(str)
    
    return data

def make_predictions(data, num_days):
    sequence = data['Close'].values  # Assuming 'Close' prices as the sequence
    sequence = np.array(sequence, dtype=np.float32).reshape( -1, 1)
    
    predictions = []
    for _ in range(num_days):
        prediction = model.predict(sequence)
        predictions.append(prediction[0, 0])
        sequence = np.append(sequence, prediction, axis=1)  # Shift the input sequence
    
    return predictions

def scale_predictions(predictions, scaler):
    scaled_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return scaled_predictions


def plot_prices(data, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Real Stock Prices', color='blue')
    predicted_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1)[1:]
    plt.plot(predicted_dates, predictions, label='Predicted Stock Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {selected_ticker} ({forecast_horizon_days})')
    plt.legend()
    return plt


def create_plot(btc_data, predictions):
    fig = px.line(btc_data['Close'], x=btc_data.index, y='Close' ,title='Real and Predicted BTC Prices for the Next 30 Days')
    predicted_dates = pd.date_range(start=btc_data.index[-1], periods=len(predictions)+1)[1:]
    fig.add_trace(px.line(x=predicted_dates, y=predictions.flatten()).data[0])
    return fig


def plot_predit_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=btc_data['Close'].index, y='Close' ,name='Real and Predicted BTC Prices for the Next 30 Days'))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)


if st.button('Get Forecast'):
    st.write('Fetching historical BTC price data...')
    btc_data = fetch_btc_data(forecast_horizon_days + 365)  # Fetch 1-year historical data
    
    if btc_data is not None:
        st.write(f'Predicting stock prices for the next {forecast_horizon_days} days for {selected_ticker}...')
        btc_data = preprocess_data(btc_data)
        num_days = forecast_horizon_days
        
        predictions = make_predictions(btc_data, num_days)
    
        scaled_predictions = scale_predictions(predictions, scaler)

        #st.write(f'Predicted BTC prices for the next {forecast_horizon_days}:')
        #for day, price in enumerate(scaled_predictions, 1):
           #st.write(f'Day {day}: {price:.2f}')


        st.write('Real-time and Predicted BTC Prices:')
        fig = create_plot(btc_data, scaled_predictions)
        st.plotly_chart(fig) 

        #plot_predit_data()


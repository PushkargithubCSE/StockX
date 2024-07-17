import streamlit as st
from datetime import date
import yfinance as yf
from neuralprophet import NeuralProphet
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "TSLA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 5)
periods = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data ...")
data = load_data(selected_stock)
data_load_state.text("Loading Data successfully done")

st.subheader('Raw data')
st.write(data.tail())

# Prepare data for NeuralProphet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = NeuralProphet()
model.fit(df_train, freq='D')
future = model.make_future_dataframe(df_train, periods=periods)
forecast = model.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

fig1 = model.plot(forecast)
st.plotly_chart(fig1)





          






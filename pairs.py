import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

# Function to get stock data
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Function to perform cointegration test
def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            stock1 = data.iloc[:, i]
            stock2 = data.iloc[:, j]
            result = coint(stock1, stock2)
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:  # Cointegration threshold
                pairs.append((data.columns[i], data.columns[j], pvalue))
    return pvalue_matrix, pairs

# Function to rank pairs using the Ornstein-Uhlenbeck process
def rank_pairs(data, pairs):
    ranked_pairs = []
    for (stock1, stock2, pvalue) in pairs:
        spread = data[stock1] - data[stock2]
        model = LinearRegression().fit(np.arange(len(spread)).reshape(-1, 1), spread)
        halflife = -np.log(2) / model.coef_[0]
        ranked_pairs.append((stock1, stock2, pvalue, halflife))
    ranked_pairs.sort(key=lambda x: x[3])
    return ranked_pairs

# Streamlit app
st.title('Pairs Trading Dashboard')

st.sidebar.header('User Input Parameters')
tickers = st.sidebar.text_input('Stock Tickers (comma separated)', 'AAPL, MSFT, GOOG, AMZN, FB, TSLA')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-01-01'))

tickers = [ticker.strip() for ticker in tickers.split(',')]
data = get_stock_data(tickers, start_date, end_date)

st.write(f'Stock Data from {start_date} to {end_date}')
st.dataframe(data.tail())

pvalue_matrix, pairs = find_cointegrated_pairs(data)
st.write('Cointegrated Pairs')
st.dataframe(pairs, columns=['Stock 1', 'Stock 2', 'P-Value'])

ranked_pairs = rank_pairs(data, pairs)
st.write('Ranked Pairs based on Ornstein-Uhlenbeck process')
st.dataframe(ranked_pairs, columns=['Stock 1', 'Stock 2', 'P-Value', 'Half-Life'])

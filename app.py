from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import datetime
import os

app = Flask(__name__)


def download_stock_data(ticker, data_folder='data'):
    # Check if the file already exists
    file_path = os.path.join(data_folder, f'{ticker}.csv')
    if os.path.exists(file_path):
        return f"Stock data for {ticker} already exists."

    # Download data from Yahoo Finance and save to CSV
    stock_data = yf.download(ticker, start='2000-01-01', end=datetime.datetime.now())
    stock_data.to_csv(file_path)
    print(f"Retrieved stock data for {ticker}.")

    return stock_data


@app.route("/")
def main():
    return "This is the Stock Prediction API."


@app.route('/fetch_data/<ticker>', methods=['GET'])
def fetch_data(ticker):
    stock_data = download_stock_data(ticker)
    return f"Fetched data for {ticker} successfully."


if __name__ == '__main__':
    app.run(debug=True)

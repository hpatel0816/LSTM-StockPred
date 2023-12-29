import yfinance as yf
import time
import uuid
import datetime
import os


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


def generate_req_tag():
    timestamp = int(time.time())  # Get current timestamp
    unique_id = uuid.uuid4().hex[:6]  # Generate a random hex string
    return f"{timestamp}_{unique_id}"
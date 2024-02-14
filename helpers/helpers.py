import yfinance as yf
from tensorflow.keras.models import load_model
import time
import uuid
import os


def download_stock_data(ticker, data_folder='data'):
    # Check if the file already exists
    file_path = os.path.join(data_folder, f'{ticker}.csv')
    if os.path.exists(file_path):
        return f"Stock data for {ticker} already exists."

    # Download data from Yahoo Finance and save to CSV
    stock_data = yf.download(ticker, start='2019-01-01', end='2023-12-31')
    stock_data.to_csv(file_path)
    print(f"Retrieved stock data for {ticker}.")

    return stock_data


def save_model(model, ticker):
    model.save(f"model_files/saved_models/{ticker}_model.keras")


def load_saved_model(ticker):
    model_path = f"model_files/saved_models/{ticker}_model.keras"
    if os.path.isfile(model_path):
        model = load_model(model_path)
        print(f"Loaded existing model for {ticker}.")
        return model
    else:
        print(f"Model file not found for {ticker}.")
        return None


def generate_req_tag():
    timestamp = int(time.time())  # Get current timestamp
    unique_id = uuid.uuid4().hex[:6]  # Generate a random hex string
    return f"{timestamp}_{unique_id}"
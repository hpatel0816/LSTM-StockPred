from flask import Flask, jsonify, send_file
import pandas as pd
from helpers.helpers import download_stock_data, generate_req_tag
from model.predictions import predict_stock_price

app = Flask(__name__)


@app.route("/")
def main():
    return "This is the Stock Prediction API."


@app.route("/fetch_data/<ticker>", methods=['GET'])
def fetch_data(ticker):
    stock_data = download_stock_data(ticker)
    return f"Fetched data for {ticker} successfully."


@app.route("/predict/<ticker>", methods=["GET"])
def predict(ticker):
    req_tag = generate_req_tag() # Generate unique request id
    predict_stock_price(ticker, req_tag)
    img_path = f"model_files/predictions/prediction.{req_tag}.png"
    return send_file(img_path, mimetype='image/png') # Return predicted opening price plot


if __name__ == '__main__':
    app.run(debug=True)

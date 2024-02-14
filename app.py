from flask import Flask, send_file
from helpers.helpers import download_stock_data, generate_req_tag
from model.predictions import predict_stock_price

app = Flask(__name__)

# The available stocks for model training and predictions
TICKERS = ["AAPL", "AMZN", "META"]


@app.route("/")
def main():
    return "This is the Stock Prediction API."


# ** IN PROGRESS **
@app.route("/fetch_data/<ticker>", methods=['GET'])
def fetch_data(ticker):
    stock_data = download_stock_data(ticker)
    return f"Fetched data for {ticker} successfully."

# Returns the predicted stock market value for a given stock ticker 
# The default timeframe is 15 days
@app.route("/predict/<ticker>/<timeframe>", methods=["GET"])
def predict(ticker, timeframe=15):
    req_tag = generate_req_tag() # Generate unique request id
    if ticker in TICKERS:
        predict_stock_price(ticker, timeframe, req_tag)
        img_path = f"model_files/predictions/prediction.{req_tag}.png"
        return send_file(img_path, mimetype='image/png') # Return predicted opening price plot

    return f"Cannot generate predictions for {ticker} stock."


if __name__ == '__main__':
    app.run(debug=True)

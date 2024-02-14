# LSTM-StockPred
LSTM based stock predictions with historical time series data

![Predictions](model_files/predictions/prediction.1707934047_0245c7.png)

### Overview
This project was aimed to provide users with more accurate stock predictions using historical trends and sentiment analysis of the stock. We created a LSTM (Long Short Term Memory) deep learning model using Tensorflow that is great at recognizing patterns in time series data. However, this created overdependence on the historical trends and the model wasn't able to adjsut to current conditions, which led to inaccurate predictions. To overcome this, we intergrated sentiment analysis of the stock, calculating the sentiment score each day and using that with the time series data to help the model better predict the prices. We had to setup up a data pipeline to automate the data collection and cleaning from the GNews API and curate a dataset of daily stock sentiment scores based on 20-30 news headlines for the specific stock. Lastly, this model was served with a simple Flask API to allow users to easily interact with it (The full-stack web app is under progress for this project).

### How it works
- Fetch stock news headlines using the data pipeline
- Perform data analysis for cleaning/organizing data for model
- Use NLTK to generate sentiment scores and produce dataset
- Retrieve stock data from Yahoo Finance API and combine with sentiment analysis
- Feed data into LSTM model and predict for the next 30 days (by default)
- Store the model evalutions, stock prediction graphs and the best performing model
- Flask API exposes endpoint for generating stock predictions

### Technologies Used
Flask, Tensorflow, Keras, NLTK, Sci-kit learn, Pandas, Numpy,  Yahoo Finance API, GNews API

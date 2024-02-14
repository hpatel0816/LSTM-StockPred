import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from helpers.helpers import save_model, load_saved_model
import plotly.graph_objects as go


def load_data(ticker):
    # Load the csv data
    df = pd.read_csv(f'data/{ticker}.csv')
    return df


def extract_features(dataframe):
    # Extract the variables for training
    data_cols = dataframe[['Open', 'Close', 'Mean', 'Max', 'Min']]
    return data_cols.astype(float)


def transform_data(scaler, training_df):
    # Scale the data
    scaler = scaler.fit(training_df)
    return scaler.transform(training_df)


def prepare_training_data(scaled_data_df, training_df, n_past, n_future):
    trainX, trainY = [], [] # Sets containing data used for prediction and the predicted values

    # Transform the data into its corresponding sets for the LSTM model
    for i in range(n_past, len(scaled_data_df) - n_future +1):
        trainX.append(scaled_data_df[i - n_past:i, 0:training_df.shape[1]])
        trainY.append(scaled_data_df[i + n_future - 1:i + n_future, 0])

    return np.array(trainX), np.array(trainY) # Convert to numpy arrays


def create_model(trainX, trainY):
    # Define the Autoencoder model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.35))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, trainX, trainY, epochs=60, batch_size=32, validation_split=0.15):
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    return history


def plot_training_history(history, req_tag):
    BASE_PATH = 'model_files/model_eval'
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig(f'{BASE_PATH}/training_history.{req_tag}.png')  # Save the plot as an image
    plt.close()


def get_training_dates(train_dates, lookback_window, n_days_for_prediction):
    # Extract the business days in the US only for predicting
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.date_range(list(train_dates)[-lookback_window], periods=n_days_for_prediction, freq=us_bd).tolist()


def get_predicted_vals(prediction, scaler, training_df):
    # Convert into the proper shape by copying the predict data 5x to perform inverse transform and retrive actual values
    prediction_copies = np.repeat(prediction, training_df.shape[1], axis=-1)

    return scaler.inverse_transform(prediction_copies)[:,0]


def plot_predictions(predict_dates, predicted_vals, df, req_tag, ticker):
    # Convert timestamp to date
    forecast_dates = [time_i.date() for time_i in predict_dates]

    forecast_df = pd.DataFrame({'Date': np.array(forecast_dates), 'Predicted Open': predicted_vals})
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

    original = df[['Date', 'Open']].copy()
    original['Date'] = pd.to_datetime(original['Date'])
    original = original.loc[original['Date'] >= '2023-05-01']

    # Set 'Date' as the index for both dataframes
    plot_data = pd.concat([original[['Date', 'Open']], forecast_df[['Date', 'Predicted Open']]])

    # Create Plotly figure
    fig = go.Figure()
    # Create a line plot using the concatenated data
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Open'], mode='lines', name='Original Open'))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Predicted Open'], mode='lines', name='Predicted Open'))
    fig.update_layout(title=f'{ticker} Stock Predictions',
                      xaxis_title='Date',
                      yaxis_title='Open Price')

    # Save the plot
    fig.write_image(f"model_files/predictions/prediction.{req_tag}.png")


def predict_stock_price(ticker, timeframe, req_tag):
    #plt.switch_backend('Agg')

    df = load_data(ticker)
    # Separate & convert dates to datatime objects for future training/plotting
    train_dates = pd.to_datetime(df['Date'])

    training_df = extract_features(df)
    scaler = StandardScaler()
    scaled_data_df = transform_data(scaler, training_df)

    n_future = 1 # Number of days we want to predict future values
    n_past = 7 # Number of previous days we use to predict future values
    trainX, trainY = prepare_training_data(scaled_data_df, training_df, n_past, n_future)

    # Load existing model if available, otherwise create a new one
    model = load_saved_model(ticker)
    if model is None:
        model = create_model(trainX, trainY)
        history = train_model(model, trainX, trainY)
        plot_training_history(history, req_tag)

        # Save the trained model
        save_model(model, ticker)

    lookback_window = int(timeframe) + 1
    n_days_for_prediction = int(timeframe) # Number of days we want to predict the values
    predict_period_dates = get_training_dates(train_dates, lookback_window, n_days_for_prediction)

    prediction = model.predict(trainX[-n_days_for_prediction:])
    y_pred_future = get_predicted_vals(prediction, scaler, training_df)
    
    plot_predictions(predict_period_dates, y_pred_future, df, req_tag, ticker)


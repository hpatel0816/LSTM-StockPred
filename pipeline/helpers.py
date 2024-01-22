import calendar
from datetime import datetime, timedelta
import os
import yfinance as yf

# function to return the next date of teh current date


def get_next_date(current_date):
    # Convert the input date string to a datetime object
    current_date = datetime.strptime(current_date, '%Y-%m-%d')

    # Calculate the next date by adding one day
    next_date = current_date + timedelta(days=1)

    # Convert the next date back to a string
    next_date_str = next_date.strftime('%Y-%m-%d')

    return next_date_str


# function to return all the dates for 3 months from the given month and year
def get_dates_next_3_months(current_year, current_month):
    today = datetime(current_year, current_month, 1)
    dates_in_next_3_months = []

    for _ in range(3):
        year = today.year
        month = today.month
        cal = calendar.monthcalendar(year, month)
        dates_in_month = [
            f"{year}-{month:02d}-{day:02d}" for week in cal for day in week if day != 0]
        dates_in_next_3_months.extend(dates_in_month)

        # Move to the next month
        today = today + timedelta(days=calendar.monthrange(year, month)[1] + 1)

    return dates_in_next_3_months


def download_stock_data(ticker, start_date, end_date, data_folder='data'):
    # Check if the file already exists
    file_path = os.path.join(data_folder, f'{ticker}.csv')
    if os.path.exists(file_path):
        return f"Stock data for {ticker} already exists."

    # Download data from Yahoo Finance and save to CSV
    stock_data = yf.download(ticker, start=start_date,
                             end=end_date)
    stock_data.to_csv(file_path)
    print(f"Retrieved stock data for {ticker}.")

    return stock_data


def check_for_csv(ticker):
    file_path = f"./data/{ticker}.csv"
    if os.path.exists(file_path):
        os.remove(file_path)

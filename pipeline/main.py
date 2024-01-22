'''##############  libraries used  #########################'''
from helpers import get_next_date, get_dates_next_3_months, check_for_csv, download_stock_data
import time
from dotenv import load_dotenv
import os
import json
import urllib.request
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
'''########################################################'''


"""##### Variables #######"""
compaies_lst = [
    "Amazon"]      # Add stock tickers to this list meta, googel, apple
tickers = ["AMZN"]
start_month = "10"                 # enter as number eg:- may ----> 05
start_year = "2021"
end_month = "12"
end_year = "2023"
"""#######################"""


"""
Creating the stock data from yahoo finance and storing all the stock opening, closing and vloume for a specific 
stock in a csv file in ./data/{ticker}.csv
"""

# daily cron job
# for new company cron job
for ticker in tickers:
    # checks if the csv file exists
    check_for_csv(ticker)
    download_stock_data(
        ticker, start_date=f"{start_year}-{start_month}-01", end_date=f"{end_year}-{end_month}-31")


# Loads the API key used to access the Gnews API for
# access to all archived new artilces for the past 50 years
load_dotenv()
api_keys = []
for i in range(7, 8):
    api_key = os.environ.get(f"API_KEY_{i}")
    api_keys.append(api_key)


"""
Parameters used to make the API calls, this includes 3 fundamental attributes including:-
        1) the Company name
        2) the time frame from which the articles are needed(start_date and end_date)
        3) the language of the articles
        4) the contry from which the new should originate
        5) Number of articles tat need to be called that belong to the specified time frame
"""
for api_key in api_keys:
    for company in compaies_lst:
        company = company
        dates = get_dates_next_3_months(int(start_year), int(start_month))
        new_data = []
        for date in dates:
            start_date = date

            """*******Parameter for the API call**********"""

            end_date = f"{get_next_date(start_date)}T00:39:25Z"
            start_date = f"{date}T00:39:25Z"

            language = "en"
            country = "us"
            no_articles = "50"
            """********************************************"""
            # add a 1 sec delay to avoid the too many requests error
            time.sleep(1)

            url = f"https://gnews.io/api/v4/search?q={company}&apikey={api_key}&from={start_date}&to={end_date}&lang={language}&country={country}&max={no_articles}"

            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode("utf-8"))
                # all the articles recieved from the API call
                articles = data["articles"]

                # if no articles on the specified date, skip the processing
                if articles == []:
                    continue
                # creats a new dataframe for that stores the data with form
                """ Ticker       Date                                           Headline    neg    neu    pos  compound
                    0  APPLE  2022-08-1  Start your journal today! Apple Support video ...  0.000  0.842  0.158    0.4574
                    1  APPLE  2022-08-1  This sweet new AirPods Pro 2 deal (with USB-C)...  0.075  0.751  0.173    0.4019
                """

                columns = ['Ticker', 'Date', 'Headline']
                news = pd.DataFrame(columns=columns)

                for i in range(len(articles)):
                    # articles[i].title
                    # extracting just the headline of the articles
                    headline = articles[i]['title']

                    # Data to add
                    data_to_add = [company, start_date,
                                   headline]

                    # Add data to DataFrame
                    news.loc[len(news)] = data_to_add

            # Impliments a sentiment analysis algorthm based on vedar_lexcon to extract the sentiment of the news headline
            analyzer = SentimentIntensityAnalyzer()
            scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

            # adding the analysis to the dataframe
            df_scores = pd.DataFrame(scores)
            news = news.join(df_scores, rsuffix='_right')

            """
            To extract more data from the sentiment analysis this code finds 
            1) the mean of all the sentiments of the news headline for the specified time frame
            2)the highest sentiment of the day
            3)the lowest sentiment of the day"""
            average_sen = round(news['compound'].mean(), 4)
            print("Average Compound:", average_sen)

            largest_sen = round(news['compound'].max(), 4)
            print("Largest Compound Value:", largest_sen)

            smallest_sen = round(news['compound'].min(), 4)
            print("Smallest Compound Value:", smallest_sen)

            # Creating a new dataframe that will store the mean,max and min sentiment over a specific time frame

            new_data.append([company, start_date,
                            average_sen, largest_sen, smallest_sen])

    old_stock_data = pd.read_csv(f'{company}.csv')

    columns = ['Ticker', 'Date', 'Mean', "Max", "Min"]
    stock_sen = pd.DataFrame(new_data, columns=columns)

    # Append stock_sen to old_stock_data
    final_df = old_stock_data.append(stock_sen, ignore_index=True)
    csv_file_path = f'{company}.csv'
    stock_sen.to_csv(csv_file_path, index=False)


main_data = pd.read_csv(f"./data/{ticker}.csv")

senti_data = "Amazon main.csv"

# Add a new column called 'Volume' and initialize it with 0
# Convert the 'Mean' column to a numeric type
main_data['Mean'] = 0
main_data['Max'] = 0
main_data['Min'] = 0


with open(senti_data, 'r') as inFile:
    for line in inFile:

        data = line.split(",")
        if data == ['Ticker', 'Date', 'Mean', 'Max', 'Min\n']:
            continue

        date = str(data[1][0:10])
        mean = float(data[2])
        max = float(data[3])
        min = float(data[4][0:-1])
        # Update the 'Mean' column where the 'Date' is '2021-01-01'
        main_data.loc[main_data['Date'] == f'{date}', 'Mean'] += mean
        main_data.loc[main_data['Date'] == f'{date}', 'Max'] += max
        main_data.loc[main_data['Date'] == f'{date}', 'Min'] += min

main_data.to_csv('.data/Amazon_rfinal.csv', index=False)

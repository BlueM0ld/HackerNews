from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import psycopg2
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Download VADER lexicon
nltk.download('vader_lexicon')
conn_params = {
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao",
    "host": "178.156.142.230",
    "port": "5432"
}

# Connect to the database
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        # Query to get the post IDs and their submission times
        query = '''
        SELECT items.title
        FROM "hacker_news"."items"
        WHERE items.title IS NOT NULL
        LIMIT 1000;
        '''

        # Execute the query and load the results into a DataFrame
        items = pd.read_sql_query(query, conn)

# Display the first few rows of the result
print(items.head())
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Assuming `items` is your DataFrame with titles
items['sentiment'] = items['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Show the first few rows with sentiment scores
print(items[['title', 'sentiment']].head(100))

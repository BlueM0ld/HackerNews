# Plot a bar chart of the top 100 most common words in the titles of posts on Hacker News

import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

# Download NLTK stopwords if you don't have them
nltk.download('stopwords')
nltk.download('punkt_tab')

# Database connection parameters
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
        result = pd.read_sql_query(query, conn)

# Display the first few rows of the result
print(result.head())

all_titles = result['title'].tolist()
# set stop words in english
stop_words = set(stopwords.words('english'))

all_words = []

# tokenize the titles
for title in all_titles:
    tokenised_words = word_tokenize(title.lower())
    words_no_punct = [word for word in tokenised_words if word.isalnum()]
    # check for stop words and remove them
    filtered_words = [word for word in words_no_punct if word not in stop_words]
    all_words.extend(filtered_words)

print(all_words)

# Count the frequency of each word

frequency_counter = Counter(all_words)

# print(type(frequency_counter))

# # limit_100 = frequency_counter.most_common(100)
# limit_100 = sorted(frequency_counter.items(), key=lambda x: x[1], reverse=True)[:100]
# print(type(limit_100))
# print(limit_100)

# freq_keys = limit_100.keys()
# freq_values = limit_100.values()

# plt.bar(limit_100)
# plt.show()

common_words = frequency_counter.most_common(100)

# Prepare data for plotting
words, counts = zip(*common_words)

# Plot the frequency of the first 100 most common words
plt.figure(figsize=(12, 8))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title('Top 100 Most Common Words in Titles')
plt.gca().invert_yaxis()  # Invert y-axis to show the most frequent words at the top
plt.show()
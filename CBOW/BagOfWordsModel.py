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

with open('./text8', 'r') as file:
    result = file.read()



# Display the first few rows of the result
print(result.head())

all_titles = result['title'].tolist()
# set stop words in english.
stop_words = set(stopwords.words('english'))

all_words = []

# tokenize the titles
for title in all_titles:
    tokenised_words = word_tokenize(title.lower())
    print("is token?", tokenised_words)
    words_no_punct = [word for word in tokenised_words if word.isalnum()]
    # check for stop words and remove them
    filtered_words = [word for word in words_no_punct if word not in stop_words]
    all_words.extend(filtered_words)

print(all_words)
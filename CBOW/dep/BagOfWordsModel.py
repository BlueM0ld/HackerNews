import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK stopwords and tokenizer if needed
nltk.download('stopwords')
nltk.download('punkt')

# Open and read the Text8 dataset
with open('./text8', 'r') as file:
    result = file.read()

# Tokenize the entire text
tokenized_words = word_tokenize(result.lower())  # Tokenize and convert to lowercase

# Set stopwords in English
stop_words = set(stopwords.words('english'))

# Remove punctuation and stopwords
filtered_words = [word for word in tokenized_words if word.isalnum() and word not in stop_words]

# Count the frequency of the words
word_counts = Counter(filtered_words)

# Get the 100 most common words
common_words = word_counts.most_common(100)

# Prepare data for plotting
words, counts = zip(*common_words)

# Plot the top 100 most common words
plt.figure(figsize=(12, 8))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title('Top 100 Most Common Words in Text8 Dataset')
plt.gca().invert_yaxis()  # Invert y-axis to show the most frequent words at the top
plt.show()

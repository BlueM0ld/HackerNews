import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import gensim.downloader as api
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK stopwords and tokenizer if needed
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# ---------------------------------------------------
# 1. Download the Text8 Dataset
# ---------------------------------------------------

# Download text8 dataset using Gensim's API
dataset = api.load("text8")

# Ensure the 'CBOW/data/' directory exists
os.makedirs('CBOW/data', exist_ok=True)

# Save the dataset to a desired path
with open("CBOW/data/text8.txt", "w") as f:
    for sentence in dataset:
        f.write(" ".join(sentence) + "\n")

# Open and read the Text8 dataset
with open('CBOW/data/text8.txt', 'r') as file:
    text8 = file.read()


logging.info(text8[:100])

# ---------------------------------------------------
# 2. Tokenizing the text
# ---------------------------------------------------

# Tokenize the entire text
tokenized_words = word_tokenize(text8.lower())  # Tokenize and convert to lowercase

# Set stopwords in English
stop_words = set(stopwords.words('english'))

# Remove punctuation and stopwords
filtered_words = [word for word in tokenized_words if word.isalnum() and word not in stop_words]

logging.info(filtered_words[:5])


# ---------------------------------------------------
# 2. Generate Training Pairs for CBOW
# ---------------------------------------------------

def generate_cbow_training_data(text, window_size):
    training_data = []
    for i, word in enumerate(text):
        # Define the context window for the current word
        start = max(0, i - window_size)
        end = min(len(text), i + window_size + 1)
        
        # Context words: all words in the window except the center word
        context_words = [text[j] for j in range(start, end) if j != i]
        target_word = text[i]  # The center word
        
        # Append (context, target) pair to training data
        training_data.append((context_words, target_word))
    return training_data



# ---------------------------------------------------
# 3. (optional ?) DataLoader for Efficient Training:
# ---------------------------------------------------



# ? DATA LOADER, WINDOW SIZE,
# BATCH SIZE, EPOCHS, LEARNING RATE, OPTIMIZER, LOSS FUNCTION, EVALUATION METRICS, VISUALIZATION, SAVING AND LOADING THE MODEL, INFERENCE, AND EXPERIMENTATION


# load it as you go
# window size 2 /3
# 
import pandas as pd
import psycopg2
pd.set_option('display.max_columns', None)
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import torch.optim as optim

# Database connection parameters
conn_params = {
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao",
    "host": "178.156.142.230",
    "port": "5432"
}

# Fetch data from database
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        query = 'SELECT items.title FROM hacker_news.items ORDER BY id ASC LIMIT 1000;'
        titles = pd.read_sql_query(query, conn)

# Concatenate titles into a single long string
titles_long = ' '.join(titles['title'].astype(str).tolist())

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    words = text.split()
    stats = Counter(words)
    words = [word for word in words if stats[word] > 5]
    return words

# Preprocess the entire dataset
corpus = preprocess(titles_long)
print(type(corpus))  # <class 'list'>
print(len(corpus))   # e.g., 2141
print(corpus[:7])    # e.g., ['a', 'to', 'startups', '<COLON>', 'the', 'of', 'google']

# Create lookup tables for words
def create_lookup_tables(words):
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]
print(type(tokens))  # <class 'list'>
print(len(tokens))   # e.g., 16,680,599
print(tokens[:7])    # e.g., [8, 5, 33, 4, 2, 6, 22]

# Calculate the average embedding from the dataset tokens
average_embedding = np.mean(tokens, axis=0)
average_embedding = torch.tensor(average_embedding, dtype=torch.float32).unsqueeze(0)
print(average_embedding) # tensor([16.7660])

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(average_embedding.shape[0], 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)                          # First hidden layer
        self.fc3 = nn.Linear(64, 32)                           # Second hidden layer
        self.output_layer = nn.Linear(32, 1)                   # Output layer (single neuron for regression)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))        # Activation for the first layer
        x = torch.relu(self.fc2(x))        # Activation for the second layer
        x = torch.relu(self.fc3(x))        # Activation for the third layer
        x = self.output_layer(x)            # Output layer produces the final score
        return x

# Initialize the model
model = NeuralNetwork()

# Load pre-trained weights
pretrained_weights = torch.load('model.pth')

# Get the state_dict of the current model
model_state_dict = model.state_dict()

# Filter out unnecessary keys (e.g., layers that don't exist in the current model)
pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_state_dict}

# Load only matching weights
model_state_dict.update(pretrained_weights)

# Load the updated state dict into the model
model.load_state_dict(model_state_dict)

# Test the specific hardcoded title
hardcoded_title = 'Woz Interview: the early days of Apple'
# Preprocess the hardcoded title
preprocessed_title = preprocess(hardcoded_title)
print(f'Preprocessed Title: {preprocessed_title}')

# Convert the preprocessed title into tokens
hardcoded_tokens = [words_to_ids.get(word, 0) for word in preprocessed_title]  # Use 0 for unknown words
print(f'Tokens for Hardcoded Title: {hardcoded_tokens}')

# Calculate the average embedding for the hardcoded title
if len(hardcoded_tokens) > 0:
    hardcoded_average_embedding = np.mean(hardcoded_tokens)
else:
    hardcoded_average_embedding = 0  # Handle case with no valid tokens

# Ensure the average_embedding is a torch tensor
hardcoded_average_embedding_tensor = torch.tensor([[hardcoded_average_embedding]], dtype=torch.float32)

# Pass the average embedding of the hardcoded title through the model to get the predicted score
predicted_score = model(hardcoded_average_embedding_tensor)

# Detach the tensor from the computation graph and convert to NumPy
output_value = predicted_score.detach().numpy()
print('output_value', output_value)  # This will give you a plain NumPy array without the gradient function

# Or, if you want just the scalar value
output_scalar = predicted_score.item()
print(output_scalar)  # This will print the scalar value, e.g., 0.1493

# Define the true score (label) for comparison
true_score = torch.tensor([[7.0]], dtype=torch.float32)  # True score for the hardcoded title

# Compute the Mean Squared Error loss
mse_loss = nn.MSELoss()
loss = mse_loss(predicted_score, true_score)

# Print the predicted score and the loss
print(f'Predicted Score: {predicted_score.item()}')  # Convert tensor to scalar
print(f'True Score: {true_score.item()}')            # Convert tensor to scalar
print(f'Mean Squared Error Loss: {loss.item()}')     # Print the loss

import pandas as pd
import psycopg2
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt

# Database connection parameters
conn_params = {
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao",
    "host": "178.156.142.230",
    "port": "5432"
}

# Fetch data from database (title and score)
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        query = 'SELECT items.title, items.score FROM hacker_news.items WHERE items.title IS NOT NULL ORDER BY id ASC LIMIT 1000;'
        data = pd.read_sql_query(query, conn)

# Concatenate titles into a single long string for preprocessing
titles_long = ' '.join(data['title'].astype(str).tolist())

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
    # words = [word for word in words if stats[word] > 5]
    return words

# Preprocess the entire dataset
corpus = preprocess(titles_long)
print(type(corpus)) # <cl
print(len(corpus))  # 8574
print(corpus[:7])   # 'y', 'combinator', 'a', "student's", 'guide', 'to', 'startups']

# Create lookup tables for words
def create_lookup_tables(words):
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

words_to_ids, ids_to_words = create_lookup_tables(corpus)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 128)  # Adjusted input dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output_layer(x)
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

# Iterate through the data to predict scores for each title
predicted_scores = []
losses = []
mse_loss = nn.MSELoss()

for index, row in data.iterrows():
    # Preprocess the title
    preprocessed_title = preprocess(row['title'])

    # Convert the preprocessed title into tokens
    hardcoded_tokens = [words_to_ids.get(word, 0) for word in preprocessed_title]

    # Calculate the average embedding for the hardcoded title
    if len(hardcoded_tokens) > 0:
        hardcoded_average_embedding = np.mean(hardcoded_tokens)
    else:
        hardcoded_average_embedding = 0

    # Ensure the average embedding is a torch tensor
    hardcoded_average_embedding_tensor = torch.tensor([[hardcoded_average_embedding]], dtype=torch.float32)

    # Pass the average embedding of the hardcoded title through the model to get the predicted score
    predicted_score = model(hardcoded_average_embedding_tensor)

    # Detach the tensor from the computation graph and convert to NumPy
    output_scalar = predicted_score.item()
    predicted_scores.append(output_scalar)

    # Compute the Mean Squared Error loss
    true_score = row['score']  # Use the true score from the DataFrame
    loss = mse_loss(predicted_score, torch.tensor([[true_score]], dtype=torch.float32))
    losses.append(loss.item())

# Add predicted scores and losses to the DataFrame
data['predicted_score'] = predicted_scores
data['mse_loss'] = losses

# Print the results
print(data[['title', 'score', 'predicted_score', 'mse_loss']])

plt.figure(figsize=(12, 6))
plt.plot(losses, label='MSE Loss', color='blue', marker='o', markersize=3)
plt.title('Mean Squared Error Loss Over Titles')
plt.xlabel('Title Index')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
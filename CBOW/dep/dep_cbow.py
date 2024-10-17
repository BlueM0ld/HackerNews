import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import gensim.downloader as api
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------
# 1. Stream the Text8 Dataset
# ---------------------------------------------------

def stream_text8():
    """
    Streams the Text8 dataset sentence by sentence.
    """
    # Load dataset in memory-efficient streaming mode
    dataset = api.load("text8")
    for sentence in dataset:
        yield sentence  # Yield one sentence at a time for tokenization

# ---------------------------------------------------
# 2. Tokenizing the text
# ---------------------------------------------------

# Download NLTK stopwords and tokenizer if needed
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Define stopwords
stop_words = set(stopwords.words('english'))


# Tokenization and filtering for each sentence
def process_sentence(sentence):
    """
    Tokenizes and filters a sentence.
    Args:
        sentence (list): A list of words (strings).
    
    Returns:
        list: A list of filtered tokens.
    """
    # Tokenize the sentence
    tokens = word_tokenize(" ".join(sentence).lower())
    
    # Remove stopwords and punctuation
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # logging.info(filtered_tokens)
    return filtered_tokens

# ---------------------------------------------------
# 3. Generate Training Pairs for CBOW
# ---------------------------------------------------

def generate_cbow_training_data(text, window_size):
    training_data = []
    
    # Iterate through the entire corpus
    for i, word in enumerate(text):
        # Define the start and end of the context window
        start = max(0, i - window_size)  # Ensure the window doesn't go below index 0
        end = min(len(text), i + window_size + 1)  # Ensure the window doesn't exceed the text length
        
        # Extract the context words (excluding the center word)
        context_words = [text[j] for j in range(start, end) if j != i]
        target_word = text[i]  # The center word
        
        # Add (context_words, target_word) pair to the training data
        training_data.append((context_words, target_word))
    
    logging.info(training_data)
    return training_data

# ---------------------------------------------------
# 4. Process and Train with Streamed Data
# ---------------------------------------------------

# Window size for CBOW
window_size = 8

# Streaming the dataset, tokenizing, and generating CBOW pairs
for sentence in stream_text8():
    tokens = process_sentence(sentence)
    
    # Generate CBOW training pairs for this sentence
    if tokens:  # Make sure we have tokens to work with
        cbow_training_pairs = generate_cbow_training_data(tokens, window_size)
        
        # Display first 5 pairs from each processed sentence (for demonstration)
        # logging.info(f"Training pairs for the sentence: {cbow_training_pairs[:5]}")

# ---------------------------------------------------
# 5. Implement the CBOW Model
# ---------------------------------------------------

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)
    
    def forward(self, context_words):
        # Get the embeddings for context words
        embedded = self.embedding(context_words)
        
        # Average the context embeddings
        context_vector = torch.mean(embedded, dim=1)
        
        # Predict the target word
        out = self.linear(context_vector)
        return out

# ---------------------------------------------------
# 6. Train the CBOW Model
# ---------------------------------------------------

def train_model(cbow_training_pairs, model, optimizer, loss_function):
    total_loss = 0
    
    # Iterate through training pairs
    for context_words, target_word in cbow_training_pairs:
        context_words_tensor = torch.tensor([context_words], dtype=torch.long)  # Add batch dimension
        target_word_tensor = torch.tensor([target_word], dtype=torch.long)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(context_words_tensor)
        
        # Compute loss
        loss = loss_function(output, target_word_tensor)
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    return total_loss

# ---------------------------------------------------
# 7. Stream, Tokenize, and Train
# ---------------------------------------------------

# Build vocabulary first
all_tokens = []
for sentence in stream_text8():
    tokens = process_sentence(sentence)
    all_tokens.extend(tokens)

vocab = Counter(all_tokens).most_common(vocab_size)
word_to_idx = {word: idx for idx, (word, _) in enumerate(vocab)}

# Hyperparameters
window_size = 8
embed_size = 100  # Embedding dimension
vocab_size = 50000  # Example vocab size, replace with actual vocab size
epochs = 5
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
model = CBOW(vocab_size, embed_size)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
# Streaming the dataset, tokenizing, generating CBOW pairs, and training the model
for epoch in range(epochs):
    epoch_loss = 0
    for sentence in stream_text8():
        tokens = process_sentence(sentence)
        
        if tokens:
            # Generate CBOW training pairs for the sentence
            cbow_training_pairs = generate_cbow_training_data(tokens, window_size)
            
            # Train the model on the generated training pairs
            epoch_loss += train_model(cbow_training_pairs, model, optimizer, loss_function)
    
    logging.info(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')


# ---------------------------------------------------
# 8. Save the Trained Model
# ---------------------------------------------------

torch.save(model.state_dict(), 'cbow_model.pth')


# ---------------------------------------------------
# (Optional) DataLoader for Efficient Training:
# ---------------------------------------------------



# ---------------------------------------------------
#
# (note) DataLoader for Efficient Training:
# ? DATA LOADER, WINDOW SIZE,
# BATCH SIZE, EPOCHS, LEARNING RATE, OPTIMIZER, LOSS FUNCTION, EVALUATION METRICS, VISUALIZATION, SAVING AND LOADING THE MODEL, INFERENCE, AND EXPERIMENTATION
# load it as you go
# window size 2 /3
#
# ---------------------------------------------------
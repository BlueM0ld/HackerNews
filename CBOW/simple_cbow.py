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
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------
# 0. Define Hyperparameters
# ---------------------------------------------------

HYPERPARAMS = {
    'embed_size': 100,         # Embedding dimension
    'window_size': 8,          # Context window size
    'batch_size': 128,         # Batch size for DataLoader
    'epochs': 5,               # Number of training epochs
    'learning_rate': 0.001,    # Learning rate for optimizer
    'max_grad_norm': 1.0,      # Gradient clipping max norm
    'min_word_freq': 1,        # Minimum frequency for words to be included in the vocabulary
    'vocab_size': 253856,      # Will be set after building the vocabulary
}

# ---------------------------------------------------
# 1. Load and Preprocess the Text8 Dataset
# ---------------------------------------------------

def load_corpus():
    """
    Loads the Text8 dataset and returns it as a list of words.
    """
    logging.info("Loading Text8 dataset...")
    dataset = api.load("text8")
    texts = []
    for sentence in dataset:
        texts.extend(sentence)
    logging.info(f"Total tokens in dataset: {len(texts)}")
    return texts

corpus = load_corpus()
# logging.info(f"Corpus: {corpus[:5]}")

# ---------------------------------------------------
# 2. Tokenizing the text
# ---------------------------------------------------

# Download NLTK stopwords and tokenizer if needed
nltk.download('stopwords')
nltk.download('punkt')

def process_corpus(tokens):
    """
    Processes raw tokens by lowercasing, tokenizing, and removing stopwords and non-alphanumeric tokens.
    
    Args:
        tokens (list): A list of raw words.
    
    Returns:
        list: A list of processed tokens.
    """
    logging.info("Processing tokens...")
    # Lowercase and tokenize
    tokens = [word.lower() for word in tokens]
    
    # Join tokens into sentences if needed (Text8 is one big sentence)
    # For simplicity, treat it as a single list of words
    
    # Remove stopwords and non-alphanumeric tokens
    filtered_tokens = [word for word in tokens if word.isalnum()]
    logging.info(f"Total tokens after processing: {len(filtered_tokens)}")
    return filtered_tokens

processed_tokens = process_corpus(corpus);
logging.info(f"Processed tokens: {type(processed_tokens)}")
logging.info(f"Processed tokens: {len(processed_tokens)}")


# ---------------------------------------------------
# 3. Build Vocabulary
# ---------------------------------------------------


from collections import Counter

def build_vocabulary(tokens: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Builds a vocabulary mapping from tokens to unique indices.

    Args:
        tokens (list of list of str): A list where each element is a list of tokens from a sentence.

    Returns:
        vocab_to_int (dict[str, int]): A dictionary mapping words to unique indices.
        int_to_vocab (dict[int, str]): A dictionary mapping indices back to words.
    """
    token_counts = Counter(tokens)

    logging.info(f"Token counts: {len(token_counts)}")

    # Sort tokens by frequency (descending) and then alphabetically for tie-breaking
    vocab = sorted(token_counts, key=lambda k: token_counts[k], reverse=True)

    # Create lookup tables with special tokens
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    vocab_to_int['<UNK>'] = len(vocab_to_int)

    return vocab_to_int, int_to_vocab


word_to_idx, idx_to_word = build_vocabulary(processed_tokens)
logging.info(f"Vocabulary Size: {len(word_to_idx)}")


# ---------------------------------------------------
# 4. Generate Training Pairs with Encoding
# ---------------------------------------------------

def encode_word(word, word_to_idx):
    """
    Encodes a word into its corresponding index.
    
    Args:
        word (str): The word to encode.
        word_to_idx (dict): Mapping from words to indices.
    
    Returns:
        int: The index of the word.
    """
    return word_to_idx.get(word, word_to_idx["<UNK>"])

def generate_cbow_training_data(tokens, window_size, word_to_idx):
    """
    Generates CBOW training pairs from tokenized text.
    
    Args:
        tokens (list): A list of tokenized words.
        window_size (int): The context window size.
        word_to_idx (dict): Mapping from words to indices.
    
    Returns:
        list of tuples: Each tuple is (context_words, target_word)
    """
    logging.info("Generating CBOW training pairs...")
    training_data = []
    for i, word in enumerate(tokens):
        # Define the start and end of the context window
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        # Extract the context words (excluding the center word)
        context = [
            encode_word(tokens[j], word_to_idx) 
            for j in range(start, end) 
            if j != i
        ]
        target = encode_word(word, word_to_idx)
        
        training_data.append((context, target))
        
    logging.info(f"Total training pairs: {len(training_data)}")
    logging.info(f"Training data: {training_data[:5]}")

    return training_data


training_data = generate_cbow_training_data(processed_tokens, HYPERPARAMS['window_size'], word_to_idx)
logging.info(f"Training data: {training_data[:5]}")


# ---------------------------------------------------
# 5. Define the CBOW Model
# ---------------------------------------------------

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.5):
        """
        Initializes the CBOW model.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of word embeddings.
            dropout (float): Dropout rate for regularization.
        """
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(embed_size, vocab_size)

        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
    
    def forward(self, context_words):
        # Get the embeddings for context words
        embedded = self.embedding(context_words)
        
        # Average the context embeddings
        context_vector = torch.mean(embedded, dim=1)
        context_vector = self.dropout(context_vector)
        

        # Predict the target word
        out = self.linear(context_vector)
        return out

# ---------------------------------------------------
# 6. Define the Custom Dataset
# ---------------------------------------------------

class CBOWDataset(Dataset):
    def __init__(self, training_pairs):
        """
        Initializes the CBOW Dataset.
        
        Args:
            training_pairs (list of tuples): Each tuple is (context_words, target_word)
        """
        self.training_pairs = training_pairs
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        context, target = self.training_pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

#Initialize Dataset and DataLoader
logging.info("Initializing Dataset and DataLoader...")
dataset = CBOWDataset(training_data)
dataloader = DataLoader(
    dataset,
    batch_size=HYPERPARAMS['batch_size'],
    shuffle=True,              # Shuffling handled here
    num_workers=0,
    pin_memory=True            # Optional: based on your setup
)

# ---------------------------------------------------
# 7. Training Function
# ---------------------------------------------------

def train_model(dataloader, model, optimizer, loss_function, device, max_grad_norm=1.0):
    """
    Trains the CBOW model for one epoch.
    
    Args:
        dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): The CBOW model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        loss_function (nn.Module): Loss function.
        device (torch.device): Device to run the training on.
        max_grad_norm (float): Maximum norm for gradient clipping.
    
    Returns:
        float: Total loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    
    for context_batch, target_batch in dataloader:
        # Move tensors to the appropriate device
        context_batch = context_batch.to(device)  # Shape: (batch_size, context_size)
        target_batch = target_batch.to(device)    # Shape: (batch_size)
        
        # Forward pass
        output = model(context_batch)            # Shape: (batch_size, vocab_size)
        
        # Compute loss
        loss = loss_function(output, target_batch)
        total_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
    
    return total_loss

# Initialize Model, Loss Function, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Training on device: {device}")
model = CBOW(HYPERPARAMS['vocab_size'], HYPERPARAMS['embed_size']).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])

# Step 6: Training Loop
logging.info("Starting training...")
for epoch in range(HYPERPARAMS['epochs']):
    epoch_loss = train_model(dataloader, model, optimizer, loss_function, device, HYPERPARAMS['max_grad_norm'])
    logging.info(f'Epoch {epoch+1}/{HYPERPARAMS["epochs"]}, Loss: {epoch_loss:.4f}')
    
    # (Optional) Validation
    # val_loss = validate_model(val_dataloader, model_val, loss_function_val, device)
    # logging.info(f'Epoch {epoch+1}/{HYPERPARAMS["epochs"]}, Validation Loss: {val_loss:.4f}')

# Save the Trained Model
torch.save(model.state_dict(), 'cbow_model.pth')
logging.info("Model saved successfully.")

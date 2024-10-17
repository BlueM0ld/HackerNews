import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, deque
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import gensim.downloader as api
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------
# 0. Define Hyperparameters
# ---------------------------------------------------

HYPERPARAMS = {
    'vocab_size': 50000,       # ? Maximum number of unique words in the vocabulary
    'embed_size': 100,         # Embedding dimension
    'window_size': 8,          # Context window size
    'batch_size': 128,         # Batch size for DataLoader
    'epochs': 5,               # Number of training epochs
    'learning_rate': 0.001,    # Learning rate for optimizer
    'max_grad_norm': 1.0,      # Gradient clipping max norm
    'min_word_freq': 1,        # Minimum frequency for words to be included in the vocabulary
    'buffer_size': 10000,      # Buffer size for shuffling in IterableDataset
}

# ---------------------------------------------------
# 1. Stream the Text8 Dataset
# ---------------------------------------------------

def stream_text8():
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
# 3. Build Vocabulary
# ---------------------------------------------------

def build_vocabulary(stream, hyperparams):
    """
    Builds a vocabulary mapping based on word frequency.
    
    Args:
        stream (generator): A generator that yields sentences (lists of words).
        hyperparams (dict): A dictionary containing hyperparameters.
    
    Returns:
        word_to_idx (dict): Mapping from words to indices.
        idx_to_word (dict): Mapping from indices to words.
    """

    all_tokens = []
    logging.info("Streaming data to build vocabulary...")
    for sentence in stream:
        tokens = process_sentence(sentence)
        all_tokens.extend(tokens)
    
    # Count word frequencies
    word_counts = Counter(all_tokens)
    
    # Apply minimum frequency threshold
    min_freq = hyperparams.get('min_word_freq', 1)
    filtered_words = [word for word, freq in word_counts.items() if freq >= min_freq]
    
    # Select the most common words up to vocab_size - 1 (reserve for <UNK>)
    most_common = Counter(filtered_words).most_common(hyperparams['vocab_size'] - 1)
    
    # Create word_to_idx mapping
    word_to_idx = {word: idx for idx, (word, _) in enumerate(most_common, start=0)}
    word_to_idx["<UNK>"] = hyperparams['vocab_size'] - 1  # Assign the last index to <UNK>
    
    # Create idx_to_word mapping
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    logging.info(f"Built vocabulary of size: {len(word_to_idx)}")
    
    return word_to_idx, idx_to_word

# ---------------------------------------------------
# 3. Generate Training Pairs with Encoding
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

def generate_cbow_training_data_encoded_gen(text, window_size, word_to_idx):
    """
    Generator that yields encoded CBOW training pairs from tokenized text.
    
    Args:
        text (list): A list of tokenized words.
        window_size (int): The context window size.
        word_to_idx (dict): Mapping from words to indices.
    
    Yields:
        tuple: A tuple (context_words, target_word).
    """
    
    # Iterate through the entire corpus
    for i, word in enumerate(text):
        # Define the start and end of the context window
        start = max(0, i - window_size)  # Ensure the window doesn't go below index 0
        end = min(len(text), i + window_size + 1)  # Ensure the window doesn't exceed the text length
        
        # Extract the context words (excluding the center word)
        context_words = [
            encode_word(text[j], word_to_idx) 
            for j in range(start, end) 
            if j != i
        ]
        #Encode the target word
        target_word = encode_word(text[i], word_to_idx) 
        
        # Encode target word
        target_word = encode_word(text[i], word_to_idx)
        
        yield (context_words, target_word)
        


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
# 6. Define the IterableDataset with Shuffling Buffer
# ---------------------------------------------------

class CBOWIterableDatasetShuffled(IterableDataset):
    def __init__(self, stream_func, window_size, word_to_idx, buffer_size=10000):
        """
        Initializes the IterableDataset with shuffling.
    
        Args:
            stream_func (generator): A generator that yields sentences (lists of words).
            window_size (int): The context window size.
            word_to_idx (dict): Mapping from words to indices.
            buffer_size (int): Size of the shuffle buffer.
        """
        self.stream_func = stream_func
        self.window_size = window_size
        self.word_to_idx = word_to_idx
        self.buffer_size = buffer_size

    def __iter__(self):
        """
        Yields:
            tuple: (context_words_tensor, target_word_tensor)
        """
        buffer = deque(maxlen=self.buffer_size)
        
        # Fill the buffer initially
        for sentence in self.stream:
            tokens = process_sentence(sentence)
            if tokens:
                for context, target in generate_cbow_training_data_encoded_gen(tokens, self.window_size, self.word_to_idx):
                    buffer.append((context, target))
                    if len(buffer) >= self.buffer_size:
                        break
            if len(buffer) >= self.buffer_size:
                break
        
        # Shuffle the buffer
        buffer = list(buffer)
        random.shuffle(buffer)
        
        # Yield from the shuffled buffer
        for context, target in buffer:
            yield (torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long))
        
        # Continue yielding from the stream without shuffling
        for sentence in self.stream:
            tokens = process_sentence(sentence)
            if tokens:
                for context, target in generate_cbow_training_data_encoded_gen(tokens, self.window_size, self.word_to_idx):
                    yield (torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long))


# ---------------------------------------------------
# 8. Training Function
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

# ---------------------------------------------------
# 9. Validation Function (Optional)
# ---------------------------------------------------

def validate_model(dataloader, model, loss_function, device):
    """
    Validates the CBOW model on the validation set.
    
    Args:
        dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): The CBOW model.
        loss_function (nn.Module): Loss function.
        device (torch.device): Device to run the validation on.
    
    Returns:
        float: Total validation loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for context_batch, target_batch in dataloader:
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            
            output = model(context_batch)
            loss = loss_function(output, target_batch)
            total_loss += loss.item()
    return total_loss

# ---------------------------------------------------
# 10. Main Training Pipeline
# ---------------------------------------------------

def main():
    # Step 1: Build Vocabulary
    logging.info("Building vocabulary...")
    word_to_idx, idx_to_word = build_vocabulary(stream_text8(), HYPERPARAMS)
    logging.info(f"Vocabulary Size: {len(word_to_idx)}")
    
    # Step 2: Initialize the IterableDataset and DataLoader with Shuffling
    logging.info("Initializing IterableDataset with shuffling buffer...")
    shuffled_iterable_dataset = CBOWIterableDatasetShuffled(
        stream_func=stream_text8(),
        window_size=HYPERPARAMS['window_size'],
        word_to_idx=word_to_idx,
        buffer_size=HYPERPARAMS['buffer_size']
    )
    iterable_dataloader = DataLoader(
        shuffled_iterable_dataset,
        batch_size=HYPERPARAMS['batch_size'],
        shuffle=False,  # Shuffling is handled within the IterableDataset
        num_workers=4,
        pin_memory=True         # Optional: based on your setup
    )
    
    # (Optional) Initialize Validation Dataset and DataLoader
    # For demonstration, we'll skip this step, but in practice, you'd want a separate validation set.
    # Example:
    # val_iterable_dataset = CBOWIterableDatasetShuffled(...)
    # val_dataloader = DataLoader(val_iterable_dataset, ...)
    
    # Step 3: Initialize Model, Loss Function, and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on device: {device}")
    model = CBOW(HYPERPARAMS['vocab_size'], HYPERPARAMS['embed_size']).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
    
    # (Optional) Initialize Validation Model, if using validation
    # model_val = CBOW(...).to(device)
    # loss_function_val = nn.CrossEntropyLoss()
    
    # Step 4: Training Loop
    logging.info("Starting training...")
    for epoch in range(HYPERPARAMS['epochs']):
        epoch_loss = train_model(iterable_dataloader, model, optimizer, loss_function, device, HYPERPARAMS['max_grad_norm'])
        logging.info(f'Epoch {epoch+1}/{HYPERPARAMS["epochs"]}, Loss: {epoch_loss:.4f}')
        
        # (Optional) Validation
        # val_loss = validate_model(val_dataloader, model_val, loss_function_val, device)
        # logging.info(f'Epoch {epoch+1}/{HYPERPARAMS["epochs"]}, Validation Loss: {val_loss:.4f}')
    
    # Step 5: Save the Trained Model
    torch.save(model.state_dict(), 'cbow_model.pth')
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    main()

# ---------------------------------------------------
#
# (note) DataLoader for Efficient Training:
# ? DATA LOADER, WINDOW SIZE,
# BATCH SIZE, EPOCHS, LEARNING RATE, OPTIMIZER, LOSS FUNCTION, EVALUATION METRICS, VISUALIZATION, SAVING AND LOADING THE MODEL, INFERENCE, AND EXPERIMENTATION
# load it as you go
# window size 2 /3
# IterableDataset
#
# ---------------------------------------------------


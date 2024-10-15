import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

# Ensure that you have nltk data downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Sample text data
text_data = """Natural language processing (NLP) is a subfield of artificial intelligence (AI) 
focused on the interaction between computers and humans through natural language. 
The ultimate objective of NLP is to read, decipher, understand, and make sense of 
human languages in a valuable way."""

# Tokenization and preprocessing
stop_words = set(stopwords.words('english'))
tokenized_text = word_tokenize(text_data.lower())
cleaned_token_text = [word for word in tokenized_text if word.isalnum() and word not in stop_words]

# Split data into training and testing sets (90/10)
data_partition = int(len(cleaned_token_text) * 0.9)
training_data = cleaned_token_text[:data_partition]
testing_data = cleaned_token_text[data_partition:]

# Vocabulary mapping
count_words = Counter(training_data)
most_common_words = [word for word, _ in count_words.most_common(20)]  # Adjusted for simplicity
word_to_idx = {word: idx for idx, word in enumerate(most_common_words, 1)}
word_to_idx["<UNK>"] = 0  # Unknown token
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Creating training pairs
window_size = 2  # Smaller window for simplicity
def create_training_pairs(tokens, window_size):
    training_pairs = []
    for i, target_word in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if i != j:  # Avoid duplicate pairs
                training_pairs.append((tokens[i], tokens[j]))
    return training_pairs

pairs = create_training_pairs(training_data, window_size)

# Custom dataset class
class SkipGramDataset(Dataset):
    def __init__(self, pairs, word_to_idx):
       
        self.pairs = pairs
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        target_word, context_word = self.pairs[index]
        target_idx = self.word_to_idx.get(target_word, self.word_to_idx["<UNK>"])
        context_idx = self.word_to_idx.get(context_word, self.word_to_idx["<UNK>"])
        return target_idx, context_idx

# Prepare DataLoader
skipgram_dataset = SkipGramDataset(pairs, word_to_idx)
dataloader = DataLoader(skipgram_dataset, batch_size=2, shuffle=True)

# SkipGram Model
class SkipGram(nn.Module):
    def __init__(self, one_hot_vector_size=21, embedding_dim=10):
        super(SkipGram, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(one_hot_vector_size, embedding_dim), 
            nn.Linear(embedding_dim, one_hot_vector_size), 
            nn.Softmax(dim=-1)  
        )
        
    def forward(self, x):
        return self.model(x)

# Initialize model, optimizer, and loss function
skipgram_model = SkipGram(one_hot_vector_size=len(word_to_idx), embedding_dim=10)
optimizer = torch.optim.SGD(skipgram_model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for target, context in dataloader:
        # Create one-hot encoded tensor for targets
        target_one_hot = F.one_hot(target, num_classes=len(word_to_idx)).float()  # Convert to float

        # Forward pass
        skipgram_model.zero_grad()
        output = skipgram_model(target_one_hot)  # Get the model output
        loss = loss_function(output, context)  # Compute the loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print average loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")


# Function to predict context words for a given target word
def predict_context_words(target_word):
    # Check if the target word is in the vocabulary
    if target_word not in word_to_idx:
        print(f"'{target_word}' not in vocabulary.")
        return

    # Create one-hot encoding for the target word
    target_idx = word_to_idx[target_word]
    target_one_hot = F.one_hot(torch.tensor(target_idx), num_classes=len(word_to_idx)).float()

    # Get the model output
    with torch.no_grad():  # No gradient tracking needed for prediction
        output = skipgram_model(target_one_hot.unsqueeze(0))  # Add batch dimension

    # Get the probabilities and find the most likely context words
    probabilities = output.squeeze(0).numpy()  # Remove the batch dimension
    predicted_indices = probabilities.argsort()[-5:][::-1]  # Get the indices of the top 5 predicted context words

    print(f"Predicted context words for '{target_word}':")
    for idx in predicted_indices:
        print(f"- {idx_to_word[idx]} (Probability: {probabilities[idx]:.4f})")

# Example of predicting context words for a specific target word
predict_context_words('language')
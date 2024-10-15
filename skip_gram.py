# train 90/10 on the corpus data
# build skip gram
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter


# Check CPU/GPU will be useful later ish
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ds = device.with_format("torch", device=device)

# load traning dataset text9 hugging face 90/10

# Load the text8 dataset getting errorrrrrrrrs
#dataset = load_dataset('ardMLX/text8')


#with open('./text8', 'r') as file:
#    text_data = file.read()


#GONNA MAKE IT SAMLL FOR THE TIME BEING 

text_data = """Natural language processing (NLP) is a subfield of artificial intelligence (AI) 
focused on the interaction between computers and humans through natural language. 
The ultimate objective of NLP is to read, decipher, understand, and make sense of 
human languages in a valuable way."""

# NLTK to tokenise
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

tokenized_text = word_tokenize(text_data.lower()) 
cleaned_token_text = [word for word in tokenized_text if word.isalnum() and word not in stop_words]

    
print(cleaned_token_text[:50])

# Split data 90/10
data_partition = int(len(cleaned_token_text)*0.9)
training_data = cleaned_token_text[:data_partition]
testing_data = cleaned_token_text[data_partition:]

print(f"training size: {len(training_data)}, Test size: {len(testing_data)}")

# Need to map vocab to number
vocab_size = 100000
count_words = Counter(training_data)

print(count_words)

most_common_words = [word for word, _ in count_words.most_common(vocab_size - 1)]

word_to_idx = {word: idx for idx, word in enumerate(most_common_words, 1)}

print(word_to_idx)

word_to_idx["<UNK>"] = 0
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print(f"VOcab size: {len(word_to_idx)}")

window_size = 7


def create_training_pairs(tokens, window_size):
    training_pairs = []
    for i, target_word in enumerate(tokens):
        #window
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        # add pairs
        for j in range(start, end):
            if i != j:  # duplicate
                training_pairs.append((tokens[i], tokens[j]))
    
    return training_pairs

pairs = create_training_pairs(training_data, window_size=window_size)

print(f"paisr --------------: {pairs}")

#need to on hot encode

word = 'understand'
def one_hotter_encode(word,word_to_idx):
    index = word_to_idx[word]
    one_hotter = torch.nn.functional.one_hot(torch.tensor(index), num_classes=len(word_to_idx)) 
    return one_hotter

one_hot_vector = one_hotter_encode(word, word_to_idx)

print(f"one hotter vector for '{word}': {one_hot_vector}")

    
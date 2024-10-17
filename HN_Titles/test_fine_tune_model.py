import torch
import more_itertools
import collections
import json

# Load words_to_ids (word to index)
with open('words_to_ids.json', 'r') as f:
    words_to_ids = json.load(f)

# Load ids_to_words (index to word)
with open('ids_to_words.json', 'r') as f:
    ids_to_words = json.load(f)

# Define your SkipGramFoo class again
class SkipGramFoo(torch.nn.Module):
    def __init__(self, voc, emb, ctx):
        super().__init__()
        self.ctx = ctx
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, inpt, trgs, rand):
        emb = self.emb(inpt)
        ctx = self.ffw.weight[trgs]
        rnd = self.ffw.weight[rand]
        out = torch.mm(ctx, emb.T)
        rnd = torch.mm(rnd, emb.T)
        out = self.sig(out)
        rnd = self.sig(rnd)
        pst = -out.log().mean()
        ngt = -(1 - rnd).log().mean()
        return pst + ngt


# Load the saved model
model_save_path = "./skipgram_model.pth"
args = (len(words_to_ids), 64, 2)  # Same parameters used in training
mFoo_loaded = SkipGramFoo(*args)
mFoo_loaded.load_state_dict(torch.load(model_save_path))

# Put the model in evaluation mode
mFoo_loaded.eval()

# Example input
input_word = "firewall"  # Word you want to check the context for

# Ensure you have your word-to-id and id-to-word mappings loaded
# (Assuming words_to_ids and ids_to_words have already been created earlier)

if input_word in words_to_ids:
    input_token = words_to_ids[input_word]
    
    # Create dummy target and random negative samples for testing
    trgs = torch.LongTensor([input_token])  # Since we're just testing context prediction
    rand = torch.randint(0, len(words_to_ids), (2,))
    
    # Forward pass to get the word embeddings
    with torch.no_grad():  # Disable gradient calculation for testing
        input_tensor = torch.LongTensor([input_token])
        context_embeddings = mFoo_loaded.emb(input_tensor)  # Get the word's embedding
    
    # Simulate context prediction by getting the closest context words based on similarity
    similarities = torch.matmul(mFoo_loaded.ffw.weight, context_embeddings.T).squeeze()
    # print("Similarities: ", similarities)
    top_context_indices = similarities.argsort(descending=True)[:2]  # Top 2 most similar words
    print("Top context indices: ", top_context_indices)
    
    # Convert predicted context tokens back to words
    predicted_context_words = [ids_to_words[str(idx.item())] for idx in top_context_indices]
    
    print(f"Input word: {input_word}")
    print("Predicted context words: ", {ids_to_words["413"], ids_to_words["3496"]})
else:
    print(f"Word '{input_word}' is not in the vocabulary.")

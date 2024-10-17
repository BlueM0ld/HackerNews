#
#
#
import tqdm
import collections
import more_itertools
import wandb
import torch


#
#
#
torch.manual_seed(42)


#
#
#
with open('text8') as f: text8: str = f.read()


#
#
#
def preprocess(text: str) -> list[str]:
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
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  return words


#
#
#
corpus: list[str] = preprocess(text8)
print(type(corpus)) # <class 'list'>
print(len(corpus))  # 16,680,599
print(corpus[:7])   # ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']


#
#
#
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+2: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  int_to_vocab[1] = '<UNK>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab


#
#
#
words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]
print(type(tokens)) # <class 'list'>
print(len(tokens))  # 16,680,599
print(tokens[:7])   # [5234, 3081, 12, 6, 195, 2, 3134]


#
#
#
print(ids_to_words[5234])        # anarchism
print(words_to_ids['anarchism']) # 5234
print(words_to_ids['have'])      # 3081
print(len(words_to_ids))         # 63,642


#
#
#
class SkipGramOne(torch.nn.Module):
  def __init__(self, voc, emb, _):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc)
    self.max = torch.nn.Softmax(dim=1)

  def forward(self, inpt, trgs):
    emb = self.emb(inpt)
    out = self.ffw(emb)
    sft = self.max(out)
    return -(sft[0, trgs]).log().mean()
    #cel in code, maths
    # custom loss function in maths




#
#
#
class SkipGramTwo(torch.nn.Module):
  def __init__(self, voc, emb, ctx):
    super().__init__()
    self.ctx = ctx
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=ctx*voc)
    self.max = torch.nn.Softmax(dim=1)

  def forward(self, inpt, trgs):
    emb = self.emb(inpt)
    hid = self.ffw(emb)
    lgt = hid.view(self.ctx, -1)
    sft = self.max(lgt)
    arg = torch.arange(sft.size(0))
    foo = sft[arg, trgs]
    return -foo.log().mean()

#
#
#
class SkipGramTre(torch.nn.Module):
  def __init__(self, voc, emb, ctx):
    super().__init__()
    self.ctx = ctx
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    self.sig = torch.nn.Sigmoid()

  def forward(self, inpt, trgs):
    emb = self.emb(inpt)
    ctx = self.ffw.weight[trgs]
    lgt = torch.mm(ctx, emb.T)
    sig = self.sig(lgt)
    return -sig.log().mean()


#
#
#
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


#
#
#
args = (len(words_to_ids), 64, 2)
mOne = SkipGramOne(*args)
mTwo = SkipGramTwo(*args)
mTre = SkipGramTre(*args)
mFoo = SkipGramFoo(*args)


#
#
#
print('mOne', sum(p.numel() for p in mOne.parameters()))
print('mTwo', sum(p.numel() for p in mTwo.parameters()))
print('mTre', sum(p.numel() for p in mTre.parameters()))
print('mFoo', sum(p.numel() for p in mFoo.parameters()))


#
#
#
opOne = torch.optim.Adam(mOne.parameters(), lr=0.003)
opTwo = torch.optim.Adam(mTwo.parameters(), lr=0.003)
opTre = torch.optim.Adam(mTre.parameters(), lr=0.003)
opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.001)


# #
# #
# #
# wandb.init(project='skip-gram', name='mOne')
# for epoch in range(10):
#   wins = more_itertools.windowed(tokens[:10000], 3)
#   prgs = tqdm.tqdm(enumerate(wins), total=len(tokens[:10000]), desc=f"Epoch {epoch+1}", leave=False)
#   for i, tks in prgs:
#     opOne.zero_grad()
#     inpt = torch.LongTensor([tks[1]])
#     trgs = torch.LongTensor([tks[0], tks[2]])
#     loss = mOne(inpt, trgs)
#     loss.backward()
#     opOne.step()
#     wandb.log({'loss': loss.item()})
# wandb.finish()


# #
# #
# #
# wandb.init(project='skip-gram', name='mTwo')
# for epoch in range(10):
#   wins = more_itertools.windowed(tokens[:10000], 3)
#   prgs = tqdm.tqdm(wins, desc=f"Epoch {epoch+1}", leave=False)
#   for i, tks in prgs:
#     inpt = torch.LongTensor([tks[1]])
#     trgs = torch.LongTensor([tks[0], tks[2]])
#     opTwo.zero_grad()
#     loss = mTwo(inpt, trgs)
#     loss.backward()
#     opTwo.step()
#     wandb.log({'loss': loss.item()})
# wandb.finish()


# #
# #
# #
# wandb.init(project='skip-gram', name='mTre')
# for epoch in range(10):
#   wins = more_itertools.windowed(tokens[:10000], 3)
#   prgs = tqdm.tqdm(enumerate(wins), total=len(tokens[:10000]), desc=f"Epoch {epoch+1}", leave=False)
#   for i, tks in prgs:
#     inpt = torch.LongTensor([tks[1]])
#     trgs = torch.LongTensor([tks[0], tks[2]])
#     opTre.zero_grad()
#     loss = mTre(inpt, trgs)
#     loss.backward()
#     opTre.step()
#     wandb.log({'loss': loss.item()})
    
# wandb.finish()


#
#
#
wandb.init(project='skip-gram', name='mFoo')
for epoch in range(10):
  wins = more_itertools.windowed(tokens[:500], 3)
  prgs = tqdm.tqdm(enumerate(wins), total=len(tokens[:500]), desc=f"Epoch {epoch+1}", leave=False)
  for i, tks in prgs:
    inpt = torch.LongTensor([tks[1]])
    trgs = torch.LongTensor([tks[0], tks[2]])
    rand = torch.randint(0, len(words_to_ids), (2,))
    opFoo.zero_grad()
    loss = mFoo(inpt, trgs, rand)
    loss.backward()
    opFoo.step()
    wandb.log({'loss': loss.item()})
    
wandb.finish()

# Save the Trained Model
# torch.save(mFoo.state_dict(), 'text8_model.pth')
print("Model trained successfully.")

torch.save(mFoo.state_dict(), './text8_model_state.pth')


# Initialize the model
vocab_size = len(words_to_ids)
embedding_dim = 64  # Should match the value used during training
model = SkipGramFoo(vocab_size, embedding_dim, 2)

# Load the trained model's state dictionary
model.load_state_dict(torch.load('text8_model_state.pth'))

# Set the model to evaluation mode
model.eval()

print("Model loaded and set to evaluation mode successfully.")

# -------------------------------------------------------------------------------------------------


# Simple test data: let's assume your model is predicting context words for a given input word
# For example, if the input word is "king", the model should predict something like ["queen", "royal"]
# Here 'words_to_ids' is a dictionary mapping words to their ids

test_word = "king"
context_words = ["queen", "royal"]  # Expected context words

# Convert words to their corresponding ids
test_word_id = torch.LongTensor([words_to_ids[test_word]])  # Input word as tensor
expected_context_ids = torch.LongTensor([words_to_ids[word] for word in context_words])

# Forward pass (without computing gradients)
with torch.no_grad():
    rand = torch.randint(0, 500, (2,))
    output = model(test_word_id, expected_context_ids, rand)
    print(output)
    converted_output = ids_to_words[output]

# Print the output for inspection
print("Model output (logits):", converted_output)

# Optionally, if the model gives logits, you can get the top predicted context words
# _, predicted_context_ids = torch.topk(output, k=len(context_words))

# Convert predicted ids back to words
# predicted_context_words = [ids_to_words[id.item()] for id in predicted_context_ids[0]]

# Print the predicted context words
# print(f"Predicted context words for '{test_word}': {predicted_context_words}")
# print(f"Expected context words: {context_words}")

#
#
#
import tqdm
import collections
import more_itertools
import wandb
import torch
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import json


#
#
#
torch.manual_seed(42)


#
#
#

# Database connection parameters
conn_params = {
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao",
    "host": "178.156.142.230",
    "port": "5432"
}

# Connect to the database
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        # Query to get the post IDs and their submission times
        query = '''
        SELECT items.title
        FROM "hacker_news"."items"
        WHERE items.title IS NOT NULL;
        '''

        # Execute the query and load the results into a DataFrame
        result = pd.read_sql_query(query, conn)
        print("Data was successfully downloaded")
    
result.to_csv('output.csv', index=False)
print("Data was successfully saved")
# You can then open it in a similar way:
with open('output.csv', encoding='utf-8') as f:
    csv_data = f.read()
# with open('text8') as f: text8: str = f.read()
print("Data was successfully opened")

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
  text = text.replace('/',  ' <SLASH> ')
  text = text.replace('\\',  ' <BACK_SLASH> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace(':',  ' <COLON> ')
  text = re.sub(r'[^A-Za-z0-9\s]', ' ', text) # remove special characters
  text = re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip() # remove single-letter words
  words = text.split()
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  print(words[:50])
  return words


#
#
#
corpus: list[str] = preprocess(csv_data)
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

# Save words_to_ids (word to index)
with open('words_to_ids.json', 'w') as f:
    json.dump(words_to_ids, f)

# Save ids_to_words (index to word)
with open('ids_to_words.json', 'w') as f:
    json.dump(ids_to_words, f)

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
mFoo = SkipGramFoo(*args)


#
#
#


print('mFoo', sum(p.numel() for p in mFoo.parameters()))


opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.003)
model_save_path = "skipgram_model.pth"

wandb.init(project='skip-gram', name='mFoo')
for epoch in range(10):
  wins = more_itertools.windowed(tokens[:1000000], 3)
  prgs = tqdm.tqdm(enumerate(wins), total=len(tokens[:10000]), desc=f"Epoch {epoch+1}", leave=False)
  for i, tks in prgs:
    inpt = torch.LongTensor([tks[1]])
    trgs = torch.LongTensor([tks[0], tks[2]])
    rand = torch.randint(0, len(words_to_ids), (2,))
    opFoo.zero_grad()
    loss = mFoo(inpt, trgs, rand)
    loss.backward()
    opFoo.step()
    wandb.log({'loss': loss.item()})
    if(loss <= 0.55):
      break
  if(loss >= 0.55):
    print("Loss is too high")
    break
torch.save(mFoo.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")
wandb.finish()

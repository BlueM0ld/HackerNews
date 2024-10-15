import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        #input lauer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context):
        target_embeds = self.embeddings(target)  
        context_embeds = self.embeddings(context) 
        
        # Calculate scores (dot product)
        scores = torch.matmul(target_embeds, context_embeds.t())
        
        return scores  # Output scores for softmax

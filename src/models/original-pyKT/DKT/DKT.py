import torch
import torch.nn as nn

class DKT(nn.Module):

    def __init__(self, num_concepts, embed_dim=128, hidden_dim=256, dropout=0.2):
        super(DKT, self).__init__()
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.interaction_embed = nn.Embedding(num_concepts * 2, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_dim, num_concepts)
        
    def forward(self, c, r):
        x = c + self.num_concepts * r # x in (B, L)
        x_embed = self.interaction_embed(x) # x_embed in (B, L, embed_dim)

        h, _ = self.lstm(x_embed) # h in (B, L, hidden_dim)
        h_shift = torch.zeros_like(h) # h_shift in (B, L, hidden_dim)
        h_shift[:, 1:, :] = h[:, :-1, :] # h_shift in (B, L, hidden_dim)

        h_shift = self.dropout(h_shift)

        logits = self.out_layer(h_shift) # logits in (B, L, num_concepts)
        
        return logits
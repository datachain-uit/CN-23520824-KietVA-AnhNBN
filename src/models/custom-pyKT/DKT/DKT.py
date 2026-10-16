import torch
import torch.nn as nn

class customDKT(nn.Module):

    def __init__(self, num_concepts, embed_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.interaction_embed = nn.Embedding(num_concepts * 2, embed_dim)
        self.lstm_layer = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_dim, num_concepts)

    def forward(self, c, r, is_repeat, train=True):
        B, L = c.shape
        x = c + self.num_concepts * r # (B, L)
        h, _ = self.lstm_layer(self.interaction_embed(x)) # h in (B, L, hidden_dim)
        h = self.dropout(h) # h in (B, L, hidden_dim)

        pos = torch.arange(L, device=c.device).unsqueeze(0).expand(B, L) # pos in (B, L)
        start = torch.where(is_repeat == 0, pos, torch.zeros_like(pos)) # start in (B, L)
        start = start.cummax(dim=1).values # start in (B, L)

        h_shift = torch.zeros_like(h) # h_shift in (B, L, hidden_dim)
        h_shift[:, 1:, :] = h[:, :-1, :] # h_shift in (B, L, hidden_dim)

        batch_idx = torch.arange(B, device=c.device).unsqueeze(1) # batch_idx in (B, 1)
        h_prev = h_shift[batch_idx, start] # h_prev in (B, L, hidden_dim)

        if train:
            logits = self.out_layer(h_prev) # logits in (B, L, num_concepts)
        else:
            logits = self.out_layer(h_shift) # logits in (B, L, num_concepts)
        
        return logits

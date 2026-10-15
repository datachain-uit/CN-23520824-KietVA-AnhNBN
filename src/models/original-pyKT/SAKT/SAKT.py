import torch
import torch.nn as nn
import copy

device = "cpu" if not torch.cuda.is_available() else "cuda"

class SAKT(nn.Module):
    def __init__(self, num_concepts, seq_len, emb_dim, num_attn_heads, dropout=0.2, num_en=2):
        super().__init__()
        self.num_concepts = num_concepts
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        self.interaction_emb = nn.Embedding(num_concepts*2, emb_dim)
        self.concept_emb = nn.Embedding(num_concepts, emb_dim)

        self.position_emb = nn.Embedding(seq_len, emb_dim)

        self.blocks = get_clones(Blocks(emb_dim, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.emb_dim, 1)

    def base_emb(self, c, r, query):
        x = c + self.num_concepts * r
        cshftemb, xemb = self.concept_emb(query), self.interaction_emb(x)

        posemb = self.position_emb(pos_encode(self.seq_len))
        xemb = xemb + posemb

        return cshftemb, xemb

    def forward(self, c, r, query):

        cshftemb, xemb = self.base_emb(c, r, query)

        for i in range(self.num_en):
            xemb = self.blocks[i](cshftemb, xemb, xemb)

        logits = self.out_layer(self.dropout_layer(xemb)).squeeze(-1)

        return logits


class Blocks(nn.Module):
    def __init__(self, emb_dim, num_attn_heads, dropout):
        super().__init__()

        self.attn = nn.MultiheadAttention(emb_dim, num_attn_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(emb_dim)

        self.FFN = transformer_FFN(emb_dim, dropout)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, q, k, v):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        causal_mask = ut_mask(seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb

class transformer_FFN(nn.Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)
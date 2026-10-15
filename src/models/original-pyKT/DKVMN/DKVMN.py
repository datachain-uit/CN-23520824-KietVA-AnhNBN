import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

class DKVMN(nn.Module):
    def __init__(self, num_concepts, dim_s, size_m, dropout=0.2):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim_s = dim_s
        self.size_m = size_m

        self.k_emb_layer = nn.Embedding(num_concepts, dim_s)
        self.Mk = nn.Parameter(torch.Tensor(size_m, dim_s))
        self.Mv0 = nn.Parameter(torch.Tensor(size_m, dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = nn.Embedding(num_concepts*2, dim_s)

        self.f_layer = nn.Linear(dim_s*2, dim_s)
        self.dropout_layer = nn.Dropout(dropout)
        self.p_layer = nn.Linear(dim_s, 1)

        self.a_layer = nn.Linear(dim_s, dim_s)
        self.e_layer = nn.Linear(dim_s, dim_s)

    def forward(self, c, r):
        B, L = c.shape

        x = c + self.num_concepts * r
        k = self.k_emb_layer(c) # (B, L, dim_s)
        v = self.v_emb_layer(x) # (B, L, dim_s)

        Mvt = self.Mv0.unsqueeze(0).repeat(B, 1, 1) # (B, size_m, dim_s)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1) # (B, L, size_m)

        # Write process
        e = torch.sigmoid(self.e_layer(v)) # (B, L, dim_s)
        a = torch.tanh(self.a_layer(v)) # (B, L, dim_s)

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0 , 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1) # (B, L+1, size_m, dim_s)

        # Read process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )

        p = self.p_layer(self.dropout_layer(f)) # (B, L, 1)
        logits = p.squeeze(-1) # (B, L)

        return logits




        

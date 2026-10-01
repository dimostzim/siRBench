"""Standalone iteration-15 CNN with jointly trained additive positional branch."""
import torch
from torch import nn

class EfficacyCNN(nn.Module):
    def __init__(self, dropout=0.2, hidden_width=32, guide_channels=32, kernel_width=5):
        super().__init__()
        if guide_channels != 32 or kernel_width != 5:
            raise ValueError('Guide architecture is fixed at G32/K5')
        if hidden_width != 32 or dropout not in (0.0, 0.1, 0.2):
            raise ValueError('Configuration outside frozen iteration-15 grid')
        G, K = guide_channels, kernel_width
        self.guide = nn.Sequential(nn.Conv1d(4,G,K,padding=K//2), nn.GELU(),
                                   nn.Conv1d(G,G,K,padding=K//2), nn.GELU())
        self.flank = nn.Sequential(nn.Conv1d(5,16,5,padding=2), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(19*G+64,hidden_width),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_width,1))
        self.linear = nn.Linear(266,1)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def pool_flank(self, x):
        x = self.flank(x)
        return torch.cat((x.mean(dim=2), x.amax(dim=2)), dim=1)

    def forward(self, guide, left, right, positional):
        x = torch.cat((self.guide(guide).flatten(1), self.pool_flank(left),
                       self.pool_flank(right)), dim=1)
        return (self.head(x) + self.linear(positional)).squeeze(-1)

    def optimizer_groups(self, weight_decay):
        return [dict(params=[p for n,p in self.named_parameters() if n.endswith('weight')],
                     weight_decay=weight_decay),
                dict(params=[p for n,p in self.named_parameters() if n.endswith('bias')],
                     weight_decay=0.0)]

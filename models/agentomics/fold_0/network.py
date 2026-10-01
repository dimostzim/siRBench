"""Single jointly trained CNN/positional-skip efficacy regressor."""
import torch
from torch import nn

class SequenceRegressor(nn.Module):
    def __init__(self, channels=16, dropout=0.1):
        super().__init__()
        self.guide3 = nn.Conv1d(4, channels, 3, padding=1)
        self.guide5 = nn.Conv1d(4, channels, 5, padding=2)
        self.flank = nn.Conv1d(5, channels, 3, padding=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(80 * channels, 1, bias=False)
        self.skip = nn.Linear(266, 1, bias=True)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, guide, left, right):
        maps = [self.activation(self.guide3(guide)),
                self.activation(self.guide5(guide)),
                self.activation(self.flank(left)),
                self.activation(self.flank(right))]
        nonlinear = torch.cat([x.flatten(1) for x in maps] +
                              [x.mean(dim=2) for x in maps], dim=1)
        positional = torch.cat([x.flatten(1) for x in (guide, left, right)], dim=1)
        return (self.readout(self.dropout(nonlinear)) + self.skip(positional)).squeeze(-1)

"""ResNet feature projection, spot attention, Spatial GCN and hop aggregation."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import attn_block, SelectItem


def _reference_convolution_initialization():
    """Advance RNG exactly as the reference's unused ConvMixer initialization.

    These layers are discarded immediately: cached ResNet never used them.
    Keeping their initialization draw sequence preserves published seeded runs
    without carrying dormant trainable modules or architecture switches.
    """
    for _ in range(3):
        nn.Conv2d(32, 32, 5, groups=32, padding="same")
        nn.Conv2d(32, 32, 5, groups=32, padding="same")
        nn.Conv2d(32, 32, 1)


class SpatialGCN(nn.Module):
    """Reference mean-neighbor convolution with ReLU and L2 normalization."""
    def __init__(self, dim=1024):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adjacency):
        neighbors = adjacency / adjacency.sum(1, keepdim=True).clamp(min=1e-8)
        combined = F.relu(self.weight.mm(neighbors.mm(features).T)).T
        return F.normalize(combined, 2, 1)


class SpatialEncoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.visual_projection = nn.Linear(2048, 1024)
        self.x_embedding = nn.Embedding(128, 1024)
        self.y_embedding = nn.Embedding(128, 1024)
        _reference_convolution_initialization()
        self.spot_attention = nn.Sequential(*[
            attn_block(1024, 16, 64, 1024, dropout) for _ in range(12)])
        self.spatial_gcn = nn.ModuleList([SpatialGCN() for _ in range(6)])
        self.hop_aggregation = nn.Sequential(nn.LSTM(1024, 1024, 2), SelectItem(0))
        # The unused ConvMixer down projection was initialized after the LSTM.
        nn.Conv2d(32, 4, 1, 1)

    def forward(self, visual_features, positions, adjacency):
        positions = positions.reshape(-1, 2)
        context = self.x_embedding(positions[:, 0]) + self.y_embedding(positions[:, 1])
        hidden = self.visual_projection(visual_features)
        hidden = self.spot_attention((hidden + context).unsqueeze(0)).squeeze(0)
        hops = []
        for convolution in self.spatial_gcn:
            hidden = convolution(hidden, adjacency)
            hops.append(hidden.unsqueeze(0))
        return self.hop_aggregation(torch.cat(hops, 0)).mean(0)

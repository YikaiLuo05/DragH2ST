"""CGDR and Gene GAT, preserving the main rebuttal's post-GAT modulation."""
import torch
from torch import nn
from torch.nn import functional as F


class ContextGating(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True),
                                 nn.Dropout(0.1), nn.Linear(512, n_genes), nn.Sigmoid())
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, hidden):
        return self.mlp(hidden) + 0.5


class GeneGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(200, 1024, bias=False)
        self.attn_l = nn.Parameter(torch.zeros(1, 1, 1, 1024))
        self.attn_r = nn.Parameter(torch.zeros(1, 1, 1, 1024))
        self.feat_drop = nn.Dropout(0.5)
        self.attn_drop = nn.Dropout(0.0)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, embeddings, adjacency):
        b, g, _ = embeddings.shape
        projected = self.W(self.feat_drop(embeddings)).view(b, g, 1, 1024)
        left = (projected * self.attn_l).sum(dim=-1)
        right = (projected * self.attn_r).sum(dim=-1)
        scores = self.leaky_relu(left.unsqueeze(-1) + right.transpose(1, 2).unsqueeze(1))
        masked = torch.where(adjacency.unsqueeze(0).unsqueeze(2) > 0,
                             scores, torch.full_like(scores, -9e15))
        attention = self.attn_drop(F.softmax(masked, dim=-1))
        weights = attention.permute(0, 2, 1, 3).reshape(b, g, g)
        values = projected.permute(0, 2, 1, 3).reshape(b, g, 1024)
        output = torch.bmm(weights, values).view(b, 1, g, 1024).permute(0, 2, 1, 3)
        return F.relu(output.mean(dim=2))


class CGDR(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.gene_embeddings = nn.Parameter(torch.randn(n_genes, 200) * 0.01)
        self.gat = GeneGAT()
        self.register_buffer("gene_adj", torch.zeros(n_genes, n_genes))
        self.context_gating = ContextGating(n_genes)

    def set_prior(self, similarity, embeddings, threshold, top_k):
        adjacency = (similarity >= threshold).float()
        _, indices = similarity.topk(min(top_k, len(similarity)), dim=1)
        mask = torch.zeros_like(similarity).scatter_(1, indices, 1.0)
        adjacency = adjacency * mask
        adjacency = (adjacency + torch.eye(len(similarity), device=similarity.device) > 0).float()
        self.gene_adj.copy_(adjacency)
        with torch.no_grad():
            self.gene_embeddings.copy_(embeddings)

    def forward(self, hidden):
        # Preserve dropout draw order: reference computes gates before the GAT.
        gates = self.context_gating(hidden)
        basis = self.gat(self.gene_embeddings.unsqueeze(0), self.gene_adj).squeeze(0)
        return (hidden @ basis.T) * gates, gates

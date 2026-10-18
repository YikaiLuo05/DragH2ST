"""DragH2ST: Spatial Encoder, CGDR and Synergistic Prediction."""
from torch import nn
import torch
import pytorch_lightning as pl
from .spatial_encoder import SpatialEncoder
from .cgdr import CGDR
from .losses import composite_loss, mean_gene_pcc


class MeanActivation(nn.Module):
    def forward(self, values):
        return torch.clamp(torch.exp(values), min=1e-5, max=1e6)


class DispersionActivation(nn.Module):
    def forward(self, values):
        return torch.clamp(nn.functional.softplus(values), min=1e-4, max=1e4)


class DragH2ST(pl.LightningModule):
    def __init__(self, cfg, genes):
        super().__init__()
        self.cfg, self.genes = cfg, list(genes)
        self.spatial_encoder = SpatialEncoder(cfg["training"]["dropout"])
        self.mean = nn.Sequential(nn.Linear(1024, len(genes)), MeanActivation())
        self.disp = nn.Sequential(nn.Linear(1024, len(genes)), DispersionActivation())
        self.pi = nn.Sequential(nn.Linear(1024, len(genes)), nn.Sigmoid())
        self.linear_head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, len(genes)))
        self.cgdr = CGDR(len(genes))

    def forward(self, visual_features, positions, adjacency):
        hidden = self.spatial_encoder(visual_features, positions, adjacency)
        linear = self.linear_head(hidden)
        regulatory, gates = self.cgdr(hidden)
        alpha = self.cfg["prediction"]["regulatory_weight"]
        prediction = (1.0 - alpha) * linear + alpha * regulatory
        extra = self.mean(hidden), self.disp(hidden), self.pi(hidden)
        return prediction, extra, hidden

    def training_step(self, batch, batch_idx):
        adjacency = batch["adjacency"].squeeze(0)
        prediction, extra, hidden = self(batch["features"].squeeze(0), batch["positions"], adjacency)
        loss, terms = composite_loss(prediction, extra, hidden, batch, self.cfg["loss"])
        self.log("total_loss", loss, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        from .metrics import compute_metrics
        prediction = self(batch["features"].squeeze(0), batch["positions"], batch["adjacency"].squeeze(0))[0]
        target = batch["expression"].squeeze(0)
        metrics = compute_metrics(prediction.cpu().numpy(), target.cpu().numpy())
        self.log("val_score", metrics["pcc_hvg50"] + metrics["pcc_heg50"], batch_size=1)
        self.log("R", mean_gene_pcc(prediction, target), batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["training"]["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

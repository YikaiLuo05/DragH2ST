"""Five active objectives from the reference training configuration."""
import torch
from torch.nn import functional as F


def mean_gene_pcc(prediction, target, eps=1e-6):
    prediction, target = prediction.float(), target.float()
    p = prediction - prediction.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)
    variance = (t ** 2).sum(0)
    denominator = torch.sqrt((p ** 2).sum(0) + eps) * torch.sqrt(variance + eps)
    correlations = torch.nan_to_num((p * t).sum(0) / (denominator + eps),
                                   nan=0.0, posinf=0.0, neginf=0.0)
    valid = variance > eps
    return correlations[valid].mean() if valid.any() else prediction.new_tensor(0.0)


def rank_loss(prediction, target):
    if prediction.shape[0] < 2:
        return prediction.new_tensor(0.0)
    log_probabilities = F.log_softmax(prediction / 2.0, dim=0)
    weights = torch.exp(torch.clamp(target, min=-10, max=10))
    denominator = weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
    per_gene = -(weights / denominator * log_probabilities).sum(0)
    valid = denominator.squeeze(0) > 1e-8
    return per_gene[valid].mean() if valid.any() else prediction.new_tensor(0.0)


def zinb_loss(counts, mean, dispersion, pi, size_factors):
    eps = 1e-10
    mean = mean * size_factors[:, None]
    t1 = torch.lgamma(dispersion + eps) + torch.lgamma(counts + 1.0) - torch.lgamma(counts + dispersion + eps)
    t2 = (dispersion + counts) * torch.log(1.0 + mean / (dispersion + eps))
    t2 = t2 + counts * (torch.log(dispersion + eps) - torch.log(mean + eps))
    nb_case = t1 + t2 - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(dispersion / (dispersion + mean + eps), dispersion)
    zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
    return torch.where(torch.le(counts, 1e-8), zero_case, nb_case).mean()


def composite_loss(prediction, extra, hidden, batch, weights):
    target = batch["expression"].reshape_as(prediction)
    adjacency = batch["adjacency"].reshape(len(prediction), len(prediction))
    normalized = F.normalize(hidden, p=2, dim=-1)
    terms = {
        "mse": F.mse_loss(prediction, target),
        "reconstruction": F.mse_loss(torch.sigmoid(torch.mm(normalized, normalized.t())), adjacency),
        "pcc": -mean_gene_pcc(prediction, target),
        "rank": rank_loss(prediction, target),
        "zinb": zinb_loss(batch["counts"].reshape_as(prediction), *extra, batch["size_factors"].reshape(-1)),
    }
    total = (weights["mse"] * terms["mse"] + weights["reconstruction"] * terms["reconstruction"]
             + weights["zinb"] * terms["zinb"] + weights["rank"] * terms["rank"]
             + weights["pcc"] * terms["pcc"])
    return total, terms

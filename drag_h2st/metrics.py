"""Gene-wise PCC and MSE, with reference HVG/HEG selection conventions."""
import warnings
import numpy as np
from scipy.stats import pearsonr


def compute_metrics(prediction, target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcc = np.array([pearsonr(prediction[:, i], target[:, i])[0] for i in range(target.shape[1])])
    hvg = np.argsort(np.var(target, axis=0))[::-1][:50]
    heg = np.argsort(np.mean(target, axis=0))[::-1][:50]
    return {"pcc_all": float(np.nanmean(pcc)), "pcc_hvg50": float(np.nanmean(pcc[hvg])),
            "pcc_heg50": float(np.nanmean(pcc[heg])), "mse_all": float(np.mean((prediction - target) ** 2)),
            "mse_hvg50": float(np.nanmean((prediction[:, hvg] - target[:, hvg]) ** 2)),
            "mse_heg50": float(np.nanmean((prediction[:, heg] - target[:, heg]) ** 2))}

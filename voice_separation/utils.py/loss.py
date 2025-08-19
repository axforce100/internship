#not used
import torch
import torch.nn as nn

def si_snr(est_source, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
    Args:
        est_source: [B, T]
        target: [B, T]
    """
    # zero-mean
    est_source = est_source - torch.mean(est_source, dim=1, keepdim=True)
    target = target - torch.mean(target, dim=1, keepdim=True)

    # projection
    s_target = torch.sum(est_source * target, dim=1, keepdim=True) * target / (
        torch.sum(target ** 2, dim=1, keepdim=True) + eps
    )

    e_noise = est_source - s_target

    ratio = (torch.sum(s_target ** 2, dim=1) + eps) / (torch.sum(e_noise ** 2, dim=1) + eps)
    si_snr_val = 10 * torch.log10(ratio + eps)
    return torch.mean(-si_snr_val)  # negative for minimization

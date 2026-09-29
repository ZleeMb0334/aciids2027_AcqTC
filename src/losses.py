"""Losses: focal BCE, paired supervised contrastive, removal-consistency."""
import torch
import torch.nn.functional as F


def focal_bce(logit, y, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0.5, p, 1 - p).clamp(1e-6, 1 - 1e-6)
    at = torch.where(y > 0.5, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    return (-at * (1 - pt) ** gamma * torch.log(pt)).mean()


def paired_supcon(z, y, tau=0.1):
    """Supervised contrastive loss on L2-normalised embeddings.

    For a real anchor the positives are the other reals in the batch.  The
    negatives are every fake -- including the fake built from the same source,
    This pairing favours manipulation cues over identity or content cues.
    """
    sim = (z @ z.t()) / tau
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    same = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    logden = torch.logsumexp(sim, dim=1)
    cnt = same.sum(1).clamp(min=1)
    lognum = (sim.masked_fill(~same, -1e9)).logsumexp(dim=1) - torch.log(cnt.float())
    loss = -(lognum - logden)
    return loss[same.sum(1) > 0].mean()


def removal_consistency(noise_branch, x_f, x_f_removed, beta=1.0):
    """L_robust: the branch must call BOTH the fake and its fingerprint-removed
    version fake, and must place them close together in embedding space."""
    h1, l1 = noise_branch(x_f)
    h2, l2 = noise_branch(x_f_removed)
    ones = torch.ones_like(l1)
    ce = F.binary_cross_entropy_with_logits(l1, ones) + \
         F.binary_cross_entropy_with_logits(l2, ones)
    drift = ((h1 - h2) ** 2).sum(1).mean()
    return ce + beta * drift, drift.detach()


def reliance_entropy_loss(w, target=0.90, eps=1e-8):
    """Exploratory Shannon-entropy hinge used by the gate_srr ablation.

    The corrected analysis uses the effective band count exp(H_2).  This older
    H_1 hinge remains in the artifact because the paper reports its negative
    result:

        L_srr = relu(tau * log B - H(w_hat)).
    """
    a = w.abs() + eps
    p = a / a.sum()
    H = -(p * torch.log(p)).sum()
    logB = torch.log(torch.tensor(float(w.numel()), device=w.device))
    return torch.relu(target * logB - H), H

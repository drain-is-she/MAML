import torch
import torch.nn.functional as F


def reconstruction_loss(reconstruction, target):
    return F.mse_loss(reconstruction, target)


def kl_divergence(mu, log_var):
    return -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )


def vae_loss(reconstruction, target, mu, log_var, beta=1.0):
    recon_loss = reconstruction_loss(reconstruction, target)
    kl_loss = kl_divergence(mu, log_var)
    total_loss = recon_loss + beta * kl_loss  # was `beta * kl_divergence` (function, not value) — crashed
    return total_loss, recon_loss, kl_loss

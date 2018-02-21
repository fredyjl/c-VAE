import torch
from torch.nn import functional as F

def loss_mse(x_recon, x):
    return F.mse_loss(x_recon, x)

def loss_kld(mu, logvar):
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

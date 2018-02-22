import torch
from torch.nn import functional as F

def loss_mse(x_recon, x):
    # loss_mse already normalizes n_frames, n_band and n_winsize
    return F.mse_loss(x_recon, x)

def loss_kld(mu, logvar, x, args):
    tmp = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    tmp /= x.size()[0] * args.spec_nband * args.contextwin_winsize
    return tmp

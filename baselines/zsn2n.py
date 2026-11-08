import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def pair_downsampler(img):
    _, c, h, w = img.shape
    f1 = torch.tensor([[[[0.0,0.5],[0.5,0.0]]]],device=img.device,dtype=img.dtype).repeat(c,1,1,1)
    f2 = torch.tensor([[[[0.5,0.0],[0.0,0.5]]]],device=img.device,dtype=img.dtype).repeat(c,1,1,1)
    return F.conv2d(img,f1,stride=2,groups=c), F.conv2d(img,f2,stride=2,groups=c)


class ZSN2NNet(nn.Module):
    def __init__(self, n_chan=1, chan_embed=48):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
    def forward(self, x):
        x = self.act(self.conv1(x)); x = self.act(self.conv2(x)); return self.conv3(x)


def loss_fn(model, noisy):
    n1,n2 = pair_downsampler(noisy)
    p1 = n1 - model(n1)
    p2 = n2 - model(n2)
    lres = 0.5*(F.mse_loss(n1,p2)+F.mse_loss(n2,p1))
    den = noisy - model(noisy)
    d1,d2 = pair_downsampler(den)
    lcon = 0.5*(F.mse_loss(p1,d1)+F.mse_loss(p2,d2))
    return lres+lcon


def train_zsn2n(noisy, epochs=2000, lr=1e-3, step_size=1500, gamma=0.5, seed=123):
    set_seed(seed)
    model = ZSN2NNet(noisy.shape[1]).to(noisy.device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    model.train()
    for _ in range(epochs):
        loss = loss_fn(model,noisy)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    model.eval()
    with torch.no_grad(): den = (noisy-model(noisy)).clamp(0,1)
    return model, den, float(loss.detach().item())

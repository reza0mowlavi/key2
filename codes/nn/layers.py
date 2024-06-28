import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(normalized_shape))

    def reciprocal_rms(self, x):
        ms = torch.sum(x**2) / self.normalized_shape  ### (...,d)
        ms = x.square().sum(-1, keepdim=True) / self.normalized_shape  ### (...,1)
        reciprocal_rms = (ms + self.eps).rsqrt()  ### (...,1)
        return reciprocal_rms  ### (...,1)

    def forward(self, x):
        reciprocal_rms = self.reciprocal_rms(x)  ### (...,1)
        result = torch.einsum("...i,i -> ...i", x * reciprocal_rms, self.gain)
        return result


class GlobalAvgPooling1D(nn.Module):
    def forward(self, x, mask=None):
        ### x: (BS, S, d)
        if mask is None:
            x = x.mean(dim=-2)  ## (BS, d)
        else:
            mask = mask.view(x.size(0), x.size(1), 1)  ## (BS, S, 1)
            x = x * mask  ## (BS, S, d)
            x = x.sum(dim=-2) / mask.sum(dim=-2)  ## (BS, d)
        return x  ## (BS, d)


def normalize(x, dim, eps=1e-12):
    var, mean = torch.var_mean(x, dim=dim, keepdim=True)
    x = (x - mean) * torch.rsqrt(var + eps)
    return x


class Normalize(nn.Module):
    def __init__(self, dim, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return normalize(x, dim=self.dim, eps=self.eps)


class NormalizationLayer(nn.Module):
    def __init__(self, input_dim, eps=1e-12):
        super().__init__()
        self.register_buffer("running_mean", torch.empty(input_dim))
        self.register_buffer("running_var", torch.empty(input_dim))
        self.register_buffer("n_samples", torch.empty([]))
        self.eps = eps
        self._reset()

    def _reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.n_samples.zero_()

    def partial_fit(self, batch):
        batch_var, batch_mean = torch.var_mean(batch, dim=0)
        batch_size = batch.size(0)

        new_mean = (self.running_mean * self.n_samples + batch_mean * batch_size) / (
            self.n_samples + batch_size
        )
        new_var = (self.running_var * self.n_samples + batch_var * batch_size) / (
            self.n_samples + batch_size
        )

        self.running_mean = new_mean
        self.running_var = new_var
        self.n_samples += batch_size

    def adapt(self, iterator, reset=False):
        if reset:
            self._reset()

        for batch in iterator:
            self.partial_fit(batch)

    def forward(self, x):
        return (x - self.running_mean) * torch.rsqrt(self.running_var + self.eps)


class PrependCLSToken(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, x):
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.concat((cls_token, x), dim=1)
        return x

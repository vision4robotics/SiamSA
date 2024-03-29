"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sesn.ses_conv import ses_max_projection


class ScaleHead(nn.Module):

    def __init__(self, out_scale=1, scales=[1], head='corr'):
        super().__init__()
        self.out_scale = out_scale
        self.scales = scales
        self.head = head

        if self.head == 'corr':
            self.corr_func = self._fast_xcorr
        if self.head == 'greedypw':
            self.scalepooling = self.pw_maxpooling
            self.corr_func = self._fast_corr_scale

    def forward(self, x, z):
        z = ses_max_projection(z)
        x = ses_max_projection(x)
        pooled = self.corr_func(x, z)
        return pooled * self.out_scale

    def pw_maxpooling(self, x, scale_dim=1):
        n_scales = x.shape[scale_dim]
        n_c = x.shape[0]
        raveled = x.view(n_c, n_scales, -1)
        zero_batch_max = raveled.max(dim=-1).values[0]

        pooled = x.max(dim=1, keepdim=True).values
        return pooled

    def _fast_corr_scale(self, z, x):
        scale = self.scales
        outsize_h = x.shape[-2] - z.shape[-2] + 1
        outsize_w = x.shape[-1] - z.shape[-1] + 1

        output = torch.zeros(x.shape[0], len(scale), outsize_h, outsize_w, device=x.device)
        for i, each_scale in enumerate(scale):
            x_rescale = self.rescale4d(x, each_scale)
            y_rescale, _ = self._fast_xcorr(z, x_rescale)
            output[:, i, ...] = self.rescale4d(
                y_rescale, outsize_w / y_rescale.shape[-1]).squeeze(1)

        out = self.scalepooling(output)
        return out, output
    
    def _fast_xcorr(self, x, z):
        batch = z.size(0)
        channel = z.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3)) 
        z = z.view(batch*channel, 1, z.size(2), z.size(3))
        out = F.conv2d(x, z,groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
        

    def rescale4d(self, x, scale, mode='bicubic', padding_mode='constant'):
        if mode == 'nearest':
            align_corners = None
        else:
            align_corners = True

        if scale == 1:
            return x

        return F.interpolate(x, scale_factor=scale, mode=mode, align_corners=align_corners)

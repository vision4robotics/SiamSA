import torch
import torch.nn as nn
from .sesn.ses_conv import SESConv_H_H_1x1

class HW_att(nn.Module):
    def __init__(self, inp, attn_drop=0.):
        super(HW_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.query_conv = SESConv_H_H_1x1(1, 1)
        self.key_conv = SESConv_H_H_1x1(1, 1)
        self.value_conv = SESConv_H_H_1x1(inp, inp)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n,c,s,h,w = x.size()
        x_pool = (self.avg_pool(x.permute(0, 3, 4, 1, 2)) + self.max_pool(x.permute(0, 3, 4, 1, 2))).permute(0, 3, 4, 1, 2)
        proj_query = self.query_conv(x_pool).view(n, h*w, -1)
        proj_key = self.key_conv(x_pool).view(n, h*w, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = self.attn_drop(attention)
        proj_value = self.value_conv(x).view(n, h*w, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(n, c, s, h, w)

        out = self.gamma * out + x
        return out

class SC_att_gen(nn.Module):
    def __init__(self, inp, attn_drop=0.):
        super(SC_att_gen, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.query_conv = SESConv_H_H_1x1(inp, inp)
        self.key_conv = SESConv_H_H_1x1(inp, inp)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n,c,s,h,w = x.size()
        x_pool = self.avg_pool(x) + self.max_pool(x)
        proj_query = self.query_conv(x_pool).view(n, s*c, -1)
        proj_key = self.key_conv(x_pool).view(n, s*c, -1)
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        attention = self.softmax(energy)
        attention = self.attn_drop(attention)
        return attention
        
class SC_att(nn.Module):
    def __init__(self, inp, reduction=8):
        super(SC_att, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.value_conv = SESConv_H_H_1x1(inp, inp)

    def forward(self, x, attention):
        n,c,s,h,w = x.size()

        proj_value = self.value_conv(x).view(n, s*c, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(n, c, s, h, w)

        out = self.gamma * out + x
        return out



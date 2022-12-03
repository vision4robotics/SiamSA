
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from .connectors import ScaleHead
from .sc_att import HW_att, SC_att, SC_att_gen
from .sesn.ses_conv import SESConv_Z2_H, SESConv_H_H, ses_max_projection


class APN(nn.Module):
    
    def __init__(self,cfg):
        super(APN, self).__init__()
        channels=cfg.TRAIN.apnchannel
        channelr=cfg.TRAIN.clsandlocchannel
        scale_head = cfg.TRAIN.scale_head
        self.connect_model = ScaleHead(scales=scale_head, head='corr')

        self.gamma1 = nn.Parameter(t.zeros(1).cuda())
        self.gamma2 = nn.Parameter(t.zeros(1).cuda())

        self.conv_shape = nn.Sequential(
                nn.Conv2d(channelr, channelr,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channelr),  
                nn.ReLU(inplace=True),
                )
        
        self.anchor=nn.Conv2d(channelr, 4,  kernel_size=3, stride=1,padding=1)

        self.conv1=nn.Sequential(
                SESConv_Z2_H(channels, channelr, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channelr),
                nn.ReLU(inplace=True),
                )
        self.conv2=nn.Sequential(
                SESConv_Z2_H(channelr, channelr, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channelr),
                nn.ReLU(inplace=True),
                )
        self.conv3=nn.Sequential(
                SESConv_Z2_H(channelr, channelr, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channelr),
                nn.ReLU(inplace=True),
                )
        self.conv4=nn.Sequential(
                SESConv_Z2_H(channelr, channelr, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channelr),
                nn.ReLU(inplace=True),
                )
        self.SC_att_gen=SC_att_gen(channelr)
        self.SC_att=SC_att(channelr)    
        self.HW_att=HW_att(channelr)
        self.add=nn.ConvTranspose3d(channelr*2, channelr,  kernel_size=1, stride=1)
        self.activate=nn.ReLU(inplace=True)

        for modules in [self.conv_shape]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
                    
    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,x,z):
        ress=self.conv1(self.xcorr_depthwise(x[0],z[0]))
        res=self.conv4(self.connect_model(self.conv2(x[1]),self.conv3(z[1])))
        SC_att_ress = self.SC_att_gen(ress)
        res_CS_cross = self.SC_att(res, SC_att_ress)
        cat_res =self.activate(self.add(t.cat((res, ress),1)))
        res = res + self.gamma1*res_CS_cross +self.gamma2*cat_res
        res = ses_max_projection(res)
        res_shape = self.conv_shape(res)
        shape_pred=self.anchor(res_shape)

        return shape_pred,res

class clsandloc(nn.Module):

    def __init__(self,cfg):
        super(clsandloc, self).__init__()
        channel=cfg.TRAIN.clsandlocchannel
        scale_head = cfg.TRAIN.scale_head
        self.connect_model = ScaleHead(scales=scale_head, head='corr')
        self.conv1=nn.Sequential(
                SESConv_Z2_H(channel, channel, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True),
                )  
        self.conv2=nn.Sequential(
                SESConv_Z2_H(channel, channel, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True),
                )
        self.conv3=nn.Sequential(
                SESConv_Z2_H(channel, channel, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True),
                )
        self.conv4=nn.Sequential(
                SESConv_Z2_H(channel, channel, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True),
                )
        self.conv5=nn.Sequential(
                SESConv_H_H(channel, channel, scale_size=1, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True),
                )  
        self.conv6=nn.Sequential(
                SESConv_H_H(channel, channel, scale_size=1, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True),
                )
        self.conv7=nn.Sequential(
                SESConv_H_H(channel, channel, scale_size=1, kernel_size=3, effective_size=3, scales=scale_head, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True),
                )

        
        self.convloc = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 4,  kernel_size=3, stride=1,padding=1),
                )
        self.convcls = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                )

        self.gamma1 = nn.Parameter(t.zeros(1).cuda())
        self.gamma2 = nn.Parameter(t.zeros(1).cuda())
        
        self.SC_att_gen=SC_att_gen(channel)
        self.SC_att=SC_att(channel)    
        self.HW_att=HW_att(channel)
        self.add=nn.ConvTranspose3d(channel*2, channel,  kernel_size=1, stride=1)
        self.activate=nn.ReLU(inplace=True)
        
        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls3=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)

        for modules in [self.convloc, self.convcls, self.cls1,
                        self.cls2,self.cls3]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)

    def forward(self, x, z, ress):
        res=self.connect_model(self.conv1(x[1]),self.conv2(z[1]))
        res = self.conv3(res)
        ress = self.conv4(ress)
        # res_HW=self.HW_att(res)
        # ress_HW=self.HW_att(ress)
        # SC_att_res=self.SC_att_gen(res_HW)
        # SC_att_ress=self.SC_att_gen(ress_HW)
        SC_att_res=self.SC_att_gen(res)
        SC_att_ress=self.SC_att_gen(ress)
        # res_SC_self = self.conv5(self.SC_att(res_HW, SC_att_res))
        # res_SC_cross = self.conv6(self.SC_att(ress_HW, SC_att_res))
        # ress_SC_self = self.conv7(self.SC_att(ress_HW, SC_att_ress))
        res_SC_self = self.conv5(self.SC_att(res, SC_att_res))
        res_SC_cross = self.conv6(self.SC_att(ress, SC_att_res))
        ress_SC_self = self.conv7(self.SC_att(ress, SC_att_ress))
        cat_res =self.activate(self.add(t.cat((res_SC_self, ress_SC_self),1)))

        res = res_SC_self + self.gamma1*res_SC_cross +self.gamma2*cat_res
        res = ses_max_projection(res)
        cls=self.convcls(res)
        cls1=self.cls1(cls)
        cls2=self.cls2(cls)
        cls3=self.cls3(cls)

        loc=self.convloc(res)

        return cls1,cls2,cls3,loc
 

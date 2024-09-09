import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import *
from .init_weights import init_weights
from .ConvNextV2 import convnext_pico, convnextv2_tiny, convnextv2_small, convnextv2_base


class SLMNet(nn.Module):
    # SLMNet
    # backbone:  convnextv2_tiny
    def __init__(self, backbone='convnextv2_tiny', mid_ch=128, bottleneck_num=2, **kwargs):
        super(SLMNet, self).__init__()      

        self.encoder=convnextv2_tiny()
        enc_dims=[96, 192, 384, 768]

        if backbone=='convnext_pico':
            self.encoder=convnext_pico()
            enc_dims=[64, 128, 256, 512]
        if backbone=='convnextv2_small':
            self.encoder=convnextv2_small()
            enc_dims=[128, 256, 512, 1024]
        if backbone=='convnextv2_base':
            self.encoder=convnextv2_base()
            enc_dims=[128, 256, 512, 1024]
        
        
        out_ch=1
        # Encoder
        self.eside1=ConvModule(enc_dims[0], mid_ch)
        self.eside2=ConvModule(enc_dims[1], mid_ch)
        self.eside3=ConvModule(enc_dims[2], mid_ch)
        self.eside4=ConvModule(enc_dims[3], mid_ch)

        # Decoder
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.slmm=SLMM(mid_ch, mid_ch)

        self.dec1=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)
        self.dec2=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)
        self.dec3=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)
        self.dec4=Decoder(c1=mid_ch, c2=mid_ch, n=bottleneck_num)

        self.dside1 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside2 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside3 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside4 = nn.Conv2d(mid_ch,out_ch,3,padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        outs = self.encoder(inputs)
        c1, c2, c3, c4 = outs
    
        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)

        cc1,cc2,cc3,cc4=self.slmm(c1,c2,c3,c4)
        # Feedback from dem
        cc12=F.interpolate(cc1, size=cc2.size()[2:], mode='bilinear', align_corners=True)
        cc13=F.interpolate(cc1, size=cc3.size()[2:], mode='bilinear', align_corners=True)
        cc14=F.interpolate(cc1, size=cc4.size()[2:], mode='bilinear', align_corners=True)

        # Decoder
        up4= c4 + cc14
        up4=self.dec4(up4)

        up3=self.upsample2(up4) + c3 + cc13
        up3=self.dec3(up3)

        up2=self.upsample2(up3) + c2 + cc12
        up2=self.dec2(up2)

        up1=self.upsample2(up2) + c1 + cc1
        up1=self.dec1(up1)

        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)
      
        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)

        return S1,S2,S3,S4, torch.sigmoid(S1),torch.sigmoid(S2),torch.sigmoid(S3),torch.sigmoid(S4)


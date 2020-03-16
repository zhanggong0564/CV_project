import torch.nn as nn
import torch.nn.functional as F
import torch

class ASPP(nn.Module):
    def __init__(self,in_ch,out_ch,rate=1):
        super(ASPP,self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,1,1,padding=0,dilation=rate,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=6*rate, dilation=6*rate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,1,padding=12*rate,dilation=12*rate,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,1,padding=18*rate,dilation=18*rate,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.branch5= nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch,out_ch,1,1,0,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch*5,out_ch,1,1,padding=0,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        b,c,h,w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature,(h,w),mode='bilinear',align_corners=True)
        feature_cat = torch.cat([conv1x1,conv3x3_1,conv3x3_2,conv3x3_3,global_feature],dim=1)
        result = self.conv_cat(feature_cat)
        return result
import torch
import torch.nn as nn

class Downblock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Downblock,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out
class upblock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(upblock,self).__init__()
        self.up =nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
        self.block = Downblock(in_ch,out_ch)
    def forward(self, x,bridge):
        up = self.up(x)
        _,_,h_d,w_d = bridge.size()
        _,_,h_u,w_u = up.size()
        diff_h = (h_d-h_u)//2
        diff_w = (w_d-w_u)//2
        bridge = bridge[:,:,diff_h:h_d-diff_h,diff_w:h_d-diff_w]
        out = torch.cat([up,bridge],1)
        out = self.block(out)
        return out
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.block1 = Downblock(1,64)
        self.pooling1 = nn.MaxPool2d(2,2)
        self.block2 = Downblock(64,128)
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.block3 = Downblock(128,256)
        self.pooling3 = nn.MaxPool2d(2, 2)
        self.block4 = Downblock(256,512)
        self.pooling4 = nn.MaxPool2d(2, 2)
        self.block5 = Downblock(512,1024)
        self.up1 = upblock(1024,512)
        self.up2 = upblock(512,256)
        self.up3 = upblock(256,128)
        self.up4 = upblock(128,64)
        self.last = nn.Conv2d(64, 2, kernel_size=1)
    def forward(self, x):
        block1 = self.block1(x)
        x =self.pooling1(block1)
        block2 = self.block2(x)
        x = self.pooling2(block2)
        block3 = self.block3(x)
        x = self.pooling3(block3)
        block4 = self.block4(x)
        x = self.pooling4(block4)
        block5 = self.block5(x)
        up1 = self.up1(block5,block4)
        up2 = self.up2(up1,block3)
        up3 =self.up3(up2,block2)
        up4 = self.up4(up3,block1)
        out = self.last(up4)
        return out

if __name__ == '__main__':
    x = torch.randn((1,1,572,572))
    unet = Unet()
    unet.eval()
    y = unet(x)
    print(y.size())
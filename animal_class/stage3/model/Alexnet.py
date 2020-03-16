import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

    def __init__(self,number_class1,number_class2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier1 = nn.Conv2d(256,number_class1,kernel_size=1)
        self.classifier2 = nn.Conv2d(256, number_class2, kernel_size=1)
    def forward(self, x):
        x = self.avgpool(self.features(x))
        x1= self.classifier1(x)
        x2 = self.classifier2(x)
        return x1,F.log_softmax(x2)
